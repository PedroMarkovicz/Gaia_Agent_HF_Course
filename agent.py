# Core Agent Logic with Basic Tool Orchestration

import re
import json
import inspect  # To potentially get tool descriptions later
import os  # Needed for checking file existence in run method
from typing import Dict, Optional, Any, Union, Callable, Tuple

# Import from the new Groq-based LLM client
from llm_client import get_llm_response, initialize_client
from tools import AVAILABLE_TOOLS
from gaia_system_prompt import GAIA_SYSTEM_PROMPT


class GaiaAgent:
    """
    A flexible agent that orchestrates tool usage based on natural language questions.

    The GaiaAgent is designed to:
    1. Parse natural language questions and determine if a tool is needed
    2. Select the most appropriate tool from available tools
    3. Generate appropriate arguments for the selected tool
    4. Execute the tool and process its results
    5. Synthesize a natural language response

    The agent uses Groq's LLM API for:
    - Tool selection and argument generation
    - Direct question answering when no tool is needed
    - Final answer synthesis when tool results are available

    Key Features:
    - Automatic tool selection based on question context
    - Flexible tool argument generation
    - Robust error handling and recovery
    - Support for forced tool usage
    - File operation handling with implicit downloads

    Example Usage:
        agent = GaiaAgent()

        # Simple question (may not need tools)
        answer = agent.run("What is the capital of France?")

        # Question requiring web search
        answer = agent.run("What were the latest AI developments?")

        # Question with task ID for file operations
        answer = agent.run("Analyze this document.", task_id="doc123")

        # Force specific tool usage
        answer = agent.run("Calculate 2+2", force_tool_name="math_tool")

    Attributes:
        tools (Dict[str, Callable]): Available tools mapping names to functions
        tool_descriptions (str): Generated descriptions of available tools for LLM
    """

    def __init__(self) -> None:
        """Initializes the GAIA Agent with available tools and ensures the LLM client is ready."""
        print("[Agent] Initializing GaiaAgent...")

        # Validate AVAILABLE_TOOLS
        if not isinstance(AVAILABLE_TOOLS, dict):
            raise TypeError(
                "AVAILABLE_TOOLS must be a dictionary mapping tool names to functions"
            )
        if not AVAILABLE_TOOLS:
            raise ValueError("No tools available. AVAILABLE_TOOLS dictionary is empty.")

        # Initialize tools and their descriptions
        self.tools: Dict[str, Callable] = AVAILABLE_TOOLS
        self.tool_descriptions: str = self._generate_tool_descriptions()
        self.system_prompt: str = GAIA_SYSTEM_PROMPT

        # Initialize the LLM client
        try:
            initialize_client()
            print("[Agent] LLM client initialized successfully.")
        except Exception as e:
            print(f"[Agent] CRITICAL ERROR during LLM client initialization: {e}")
            raise RuntimeError(f"Agent failed to initialize LLM client: {e}") from e

        print(f"[Agent] Registered tools: {list(self.tools.keys())}")

    def _generate_tool_descriptions(self) -> str:
        """Generates a string describing available tools for the LLM prompt."""
        descriptions = []
        for name, func in self.tools.items():
            docstring = inspect.getdoc(func)
            description = f"- {name}: {docstring.splitlines()[0] if docstring else 'No description available.'}"
            # Add argument hints if possible (basic example)
            try:
                sig = inspect.signature(func)
                arg_details = []
                for param_name, param in sig.parameters.items():
                    param_type = (
                        param.annotation
                        if param.annotation != inspect.Parameter.empty
                        else "any"
                    )
                    # Represent type hints more simply for the prompt
                    if param_type == "any":
                        type_str = "any"
                    else:
                        type_str = str(param_type)
                        if "typing" in type_str:
                            type_str = re.sub(r"typing\.", "", type_str)
                        type_str = re.sub(r"<class '([\w\.]+)'>", r"\1", type_str)

                    arg_details.append(f"{param_name}: {type_str}")
                if arg_details:
                    description += f" (Args: {', '.join(arg_details)})"
            except Exception as e:
                print(f"[Agent] Warning: Could not get signature for tool {name}: {e}")
                pass  # Ignore errors in signature introspection for now
            descriptions.append(description)
        return "\n".join(descriptions)

    def _get_tool_args_via_llm(
        self, question: str, task_id: Optional[str], tool_name_to_use: str
    ) -> Optional[Dict[str, Any]]:
        """
        Helper function to ask LLM to generate arguments for a specific tool.

        Args:
            question: The user question.
            task_id: Optional GAIA task ID.
            tool_name_to_use: Name of the tool to generate arguments for.

        Returns:
            Dictionary of tool arguments if successful, None if failed.
        """
        prompt = (
            "You are an AI assistant. Your task is to determine the correct arguments for a given tool "
            "based on a user question and available tools. Respond ONLY with a JSON object containing the arguments.\n\n"
            f"Available tools:\n{self.tool_descriptions}\n\n"
            f"User Question: {question}\n"
            f"Task ID (use if tool needs it, e.g., for file download): {task_id}\n"
            f"Tool to Use: {tool_name_to_use}\n\n"
            "**Instructions for Argument Generation**:\n"
            "1. Analyze the question and the specified tool ({tool_name_to_use}).\n"
            "2. Determine the necessary arguments based on the tool's description and the question's details.\n"
            "3. For search tools (`web_search`, `wiki_search`, `arvix_search`), the `query` argument MUST be specific and concise.\n"
            "4. For `download_gaia_file`, the `task_id` argument should be the provided Task ID.\n"
            "5. For math tools like `multiply` or `add`, ensure both arguments `a` and `b` are provided.\n"
            "6. For `analyze_file_content`, the `file_path` argument is usually provided by the `download_gaia_file` tool first.\n\n"
            "**Output Format**:\n"
            "- Respond ONLY with a single JSON object containing the tool arguments.\n"
            "- Do NOT include any additional text, explanation, markdown formatting, or apologies.\n"
            "- The JSON object should map argument names to their values.\n"
            '- Example for web_search: { "query": "population trend Capital City X last N years" }\n'
            '- Example for download_gaia_file: { "task_id": "TASK123" }\n'
            '- Example for multiply: { "a": 5, "b": 10 }\n\n'
            f"Provide the JSON arguments for the tool {tool_name_to_use} based on the question."
        )

        print(f"[Agent] Asking LLM for arguments for tool {tool_name_to_use}...")

        try:
            # Use low temperature for deterministic argument generation
            llm_args_response = get_llm_response(
                prompt,
                temperature=0.01,
                max_new_tokens=200,
                system_prompt=self.system_prompt,
            )
            print(
                f"[Agent] LLM raw arguments response for {tool_name_to_use}: {llm_args_response}"
            )

            # Clean and parse the response
            cleaned_args_response = re.sub(
                r"^```json\n?|\n?```$", "", llm_args_response
            ).strip()
            if not cleaned_args_response:
                print(
                    f"[Agent] LLM returned empty string for tool args for {tool_name_to_use}."
                )
                return None

            tool_args = json.loads(cleaned_args_response)

            if not isinstance(tool_args, dict):
                print(
                    f"[Agent] LLM response for tool args for {tool_name_to_use} was not a valid JSON dict: {cleaned_args_response}"
                )
                return None

            # Basic validation: Check if required args are present
            if tool_name_to_use == "download_gaia_file" and "task_id" not in tool_args:
                print(
                    f"[Agent] Warning: LLM failed to provide `task_id` for `download_gaia_file`. Args received: {tool_args}"
                )
                if task_id:  # If we have the task_id contextually, inject it
                    tool_args["task_id"] = task_id
                    print("[Agent] Injected task_id contextually.")
                else:
                    return None

            print(f"[Agent] Parsed tool args for {tool_name_to_use}: {tool_args}")
            return tool_args

        except json.JSONDecodeError as e:
            print(
                f"[Agent] Failed to parse LLM tool args JSON for {tool_name_to_use}: {e}. Response: {llm_args_response}"
            )
            # Attempt to extract JSON from within potential explanations
            match = re.search(r"\{.*?\}", llm_args_response, re.DOTALL)
            if match:
                try:
                    extracted_json_str = match.group(0)
                    tool_args = json.loads(extracted_json_str)
                    if isinstance(tool_args, dict):
                        print(
                            f"[Agent] Successfully extracted JSON args after initial parse failure: {tool_args}"
                        )
                        return tool_args
                    else:
                        print("[Agent] Extracted content was not a dict.")
                        return None
                except json.JSONDecodeError:
                    print("[Agent] Extraction attempt also failed to parse JSON.")
                    return None
            else:
                print("[Agent] No JSON object found in the response for extraction.")
                return None
        except Exception as e:
            print(
                f"[Agent] Unexpected error processing tool args for {tool_name_to_use}: {e}"
            )
            return None

    def _apply_chain_of_thought(self, question: str) -> Dict[str, Any]:
        """
        Applies enhanced chain-of-thought reasoning with advanced question analysis.

        Args:
            question: The user's question

        Returns:
            Dictionary containing reasoning steps and recommended approach
        """
        prompt = (
            "You are an AI assistant with advanced reasoning capabilities. Analyze the following question thoroughly.\n\n"
            "Question Categories:\n"
            "- factual: Requires retrieving specific facts\n"
            "- mathematical: Involves calculations or numerical operations\n"
            "- temporal: Involves time-based analysis or historical data\n"
            "- comparative: Requires comparing multiple items\n"
            "- logical: Involves deductive or inductive reasoning\n"
            "- analytical: Requires data analysis or pattern recognition\n"
            "- hypothetical: Involves conditional or speculative reasoning\n"
            "- linguistic_puzzle: Involves wordplay, string manipulation, or pattern matching\n\n"
            "Analysis Requirements:\n"
            "1. Question Classification\n"
            "   - Primary category and subcategories\n"
            "   - Complexity level (simple, moderate, complex)\n"
            "   - Required reasoning types\n\n"
            "2. Information Requirements\n"
            "   - Core facts or data needed\n"
            "   - Context requirements\n"
            "   - Potential ambiguities\n\n"
            "3. Solution Strategy\n"
            "   - Step-by-step solution path\n"
            "   - Required tools or operations\n"
            "   - Verification requirements\n\n"
            "4. Edge Cases\n"
            "   - Potential ambiguities\n"
            "   - Required validations\n"
            "   - Fallback strategies\n\n"
            "Format your response as a JSON object with the following structure:\n"
            "{\n"
            '  "question_type": {\n'
            '    "primary": "factual|mathematical|temporal|comparative|logical|analytical|hypothetical|linguistic_puzzle",\n'
            '    "secondary": ["category1", "category2"],\n'
            '    "complexity": "simple|moderate|complex"\n'
            "  },\n"
            '  "reasoning_types": ["deductive", "inductive", "analytical", ...],\n'
            '  "information_needs": {\n'
            '    "core_facts": ["fact1", "fact2"],\n'
            '    "context": ["context1", "context2"],\n'
            '    "ambiguities": ["ambiguity1", "ambiguity2"]\n'
            "  },\n"
            '  "solution_steps": ["step1", "step2"],\n'
            '  "tools_needed": ["tool1", "tool2"],\n'
            '  "preferred_tool_type": "internal|external|any",\n'
            '  "verification_steps": ["verify1", "verify2"],\n'
            '  "edge_cases": ["case1", "case2"],\n'
            '  "fallback_strategy": "description"\n'
            "}\n\n"
            "Example 1:\n"
            "Q: How many movies did Christopher Nolan direct between 2010 and 2020?\n"
            "{\n"
            '  "question_type": {\n'
            '    "primary": "temporal",\n'
            '    "secondary": ["factual", "quantitative"],\n'
            '    "complexity": "moderate"\n'
            "  },\n"
            '  "reasoning_types": ["temporal_filtering", "counting"],\n'
            '  "information_needs": {\n'
            '    "core_facts": ["Complete Nolan filmography", "Release dates"],\n'
            '    "context": ["Definition of directed vs produced"],\n'
            '    "ambiguities": ["Movies released exactly on period boundaries"]\n'
            "  },\n"
            '  "solution_steps": [\n'
            '    "Get complete filmography",\n'
            '    "Filter for director role only",\n'
            '    "Filter date range 2010-2020",\n'
            '    "Count filtered results"\n'
            "  ],\n"
            '  "tools_needed": ["web_search", "wiki_search"],\n'
            '  "preferred_tool_type": "external",\n'
            '  "verification_steps": [\n'
            '    "Cross-reference multiple sources",\n'
            '    "Verify director vs producer roles"\n'
            "  ],\n"
            '  "edge_cases": [\n'
            '    "Movies released in 2010 or 2020",\n'
            '    "Co-directed movies"\n'
            "  ],\n"
            '  "fallback_strategy": "Use most reliable single source if cross-reference fails"\n'
            "}\n\n"
            "Example 2:\n"
            "Q: What is the reverse of the word 'stressed'?\n"
            "{\n"
            '  "question_type": {\n'
            '    "primary": "linguistic_puzzle",\n'
            '    "secondary": ["string_manipulation"],\n'
            '    "complexity": "simple"\n'
            "  },\n"
            '  "reasoning_types": ["reversal"],\n'
            '  "information_needs": {\n'
            '    "core_facts": ["The word to be reversed is stressed"],\n'
            '    "context": [],\n'
            '    "ambiguities": []\n'
            "  },\n"
            '  "solution_steps": [\n'
            "    \"Take the input string 'stressed'\",\n"
            '    "Reverse the string"\n'
            "  ],\n"
            '  "tools_needed": ["string_operation"],\n'
            '  "preferred_tool_type": "internal",\n'
            '  "verification_steps": ["Check if the reversed string is \'desserts\'"],\n'
            '  "edge_cases": [],\n'
            '  "fallback_strategy": "Directly answer if tool fails"\n'
            "}\n\n"
            f"Now analyze this question: {question}"
        )

        try:
            # Use very low temperature for consistent reasoning
            response = get_llm_response(
                prompt,
                temperature=0.01,
                max_new_tokens=800,
                system_prompt=self.system_prompt,
            )
            # Extract JSON from response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                reasoning = json.loads(match.group(0))
                print(f"[Agent] Enhanced chain-of-thought reasoning: {reasoning}")
                return reasoning
            else:
                print("[Agent] Failed to extract reasoning JSON from response")
                return {
                    "question_type": {
                        "primary": "unknown",
                        "secondary": [],
                        "complexity": "unknown",
                    },
                    "reasoning_types": [],
                    "information_needs": {
                        "core_facts": [],
                        "context": [],
                        "ambiguities": [],
                    },
                    "solution_steps": [],
                    "tools_needed": [],
                    "preferred_tool_type": "any",
                    "verification_steps": [],
                    "edge_cases": [],
                    "fallback_strategy": "use direct answer",
                }
        except Exception as e:
            print(f"[Agent] Error in enhanced chain-of-thought reasoning: {e}")
            return {
                "question_type": {
                    "primary": "unknown",
                    "secondary": [],
                    "complexity": "unknown",
                },
                "reasoning_types": [],
                "information_needs": {
                    "core_facts": [],
                    "context": [],
                    "ambiguities": [],
                },
                "solution_steps": [],
                "tools_needed": [],
                "preferred_tool_type": "any",
                "verification_steps": [],
                "edge_cases": [],
                "fallback_strategy": "use direct answer",
            }

    def _get_tool_decision(
        self,
        question: str,
        task_id: Optional[str],
        force_tool_name: Optional[str] = None,
        failed_tools: set = None,
    ) -> Optional[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Enhanced tool selection with advanced decision-making capabilities.
        Uses chain-of-thought reasoning and multi-criteria analysis.

        Args:
            question: The user question.
            task_id: Optional GAIA task ID.
            force_tool_name: If provided, forces the use of this tool.
            failed_tools: A set of tool names that have already failed for this question.

        Returns:
            Dictionary with tool name and arguments if successful, None if failed.
        """
        # First apply chain-of-thought reasoning
        reasoning = self._apply_chain_of_thought(question)

        decision = None
        if force_tool_name:
            print(f"[Agent] Attempting to use FORCED tool: {force_tool_name}")
            if force_tool_name not in self.tools:
                print(
                    f"[Agent] Error: Forced tool {force_tool_name} is not an available tool."
                )
                return None

            # Get arguments for the forced tool
            tool_args = self._get_tool_args_via_llm(question, task_id, force_tool_name)
            if tool_args is not None and isinstance(tool_args, dict):
                decision = {"tool_name": force_tool_name, "tool_args": tool_args}
                print(f"[Agent] Successfully got args for forced tool: {decision}")
            else:
                print(
                    f"[Agent] Could not get valid arguments for forced tool {force_tool_name}."
                )
                return None
        else:
            # Enhanced prompt using comprehensive reasoning analysis
            prompt = (
                "You are an AI assistant with advanced tool selection capabilities. "
                "Analyze the question and reasoning to select the optimal tool and construct its arguments.\n\n"
                f"Available tools:\n{self.tool_descriptions}\n\n"
                f"Previously failed tools (avoid these if possible): {list(failed_tools) if failed_tools else 'None'}\n\n"
                "Question Analysis:\n"
                f"Type: {json.dumps(reasoning['question_type'], indent=2)}\n"
                f"Reasoning Types: {', '.join(reasoning['reasoning_types'])}\n"
                f"Information Needs: {json.dumps(reasoning['information_needs'], indent=2)}\n"
                f"Solution Steps: {', '.join(reasoning['solution_steps'])}\n"
                f"Preferred Tool Type: {reasoning.get('preferred_tool_type', 'any')}\n"
                f"Edge Cases: {', '.join(reasoning['edge_cases'])}\n"
                f"Verification Steps: {', '.join(reasoning['verification_steps'])}\n\n"
                f"User Question: {question}\n"
                f"Task ID (use only if relevant for file download): {task_id}\n\n"
                "Tool Selection Framework:\n\n"
                "1. Primary Selection Criteria\n"
                "   - Tool capability match with question type\n"
                "   - Information retrieval requirements\n"
                "   - Processing capabilities needed\n"
                "   - **Adherence to Preferred Tool Type**: Prioritize 'internal' tools (math, string_operation) if specified. Use 'external' tools (web_search, wiki_search) for information retrieval.\n"
                "   - **Avoid Failed Tools**: Do not select a tool from the 'failed_tools' list unless there are no other options.\n\n"
                "2. Tool Categories and Use Cases\n"
                "   - Information Retrieval (External):\n"
                "     * web_search: Current/general information\n"
                "     * wiki_search: Historical/encyclopedic facts\n"
                "     * arvix_search: Academic/scientific information\n"
                "   - File Operations (Internal/External):\n"
                "     * download_gaia_file: Access task-specific files\n"
                "     * analyze_file_content: Process downloaded files\n"
                "   - Mathematical Operations (Internal):\n"
                "     * All math tools for explicit calculations\n"
                "   - Local Logic and NLP (Internal):\n"
                "     * string_operation: For reversing, changing case, etc.\n"
                "     * get_antonym: For finding opposites of words.\n"
                "     * match_pattern: For regex-based text searching.\n"
                "   - Specialized Processing:\n"
                "     * Image analysis, chess analysis, etc.\n\n"
                "3. Edge Case Considerations\n"
                "   - Handle ambiguous queries with verification\n"
                "   - Consider tool combinations for complex cases\n"
                "   - Plan for potential tool failures\n\n"
                "4. No-Tool Scenarios\n"
                "   - Direct factual answers\n"
                "   - Simple logical deductions\n"
                "   - Pre-known common knowledge\n\n"
                "Output Format Requirements:\n"
                "- Respond ONLY with a single JSON object\n"
                "- No additional text or explanations\n"
                '- Tool selected: { "tool_name": "<tool_name>", "tool_args": { "<arg1>": <value1>, ... } }\n'
                '- No tool needed: { "tool_name": "none" }\n\n'
                "Tool Selection Guidelines:\n"
                "1. Prefer specialized tools over general ones when applicable\n"
                "2. Consider verification requirements in tool selection\n"
                "3. Account for edge cases in argument construction\n"
                "4. Include fallback options in complex cases\n"
                "5. Validate argument completeness\n\n"
                "Make your tool selection decision now."
            )

            print("[Agent] Making enhanced tool decision...")

            try:
                # Use very low temperature for consistent tool selection
                llm_response = get_llm_response(
                    prompt,
                    temperature=0.01,
                    max_new_tokens=400,
                    system_prompt=self.system_prompt,
                )
                print(f"[Agent] LLM raw tool decision response: {llm_response}")

                # Clean and parse the response
                cleaned_response = re.sub(
                    r"^```json\n?|\n?```$", "", llm_response
                ).strip()
                if not cleaned_response:
                    print("[Agent] LLM returned empty string for tool decision.")
                    return None

                decision_candidate = json.loads(cleaned_response)

                if (
                    isinstance(decision_candidate, dict)
                    and "tool_name" in decision_candidate
                ):
                    tool_name_candidate = decision_candidate["tool_name"]

                    # Enhanced validation based on reasoning
                    if tool_name_candidate == "none":
                        # Verify that 'none' is appropriate based on reasoning
                        if (
                            reasoning["question_type"]["complexity"] == "simple"
                            and not reasoning["tools_needed"]
                            and not reasoning["edge_cases"]
                        ):
                            decision = {"tool_name": "none"}
                        else:
                            print(
                                "[Agent] 'none' selected but question appears to need tools. Attempting recovery..."
                            )
                            # Attempt to recover by selecting most appropriate tool
                            for tool in reasoning["tools_needed"]:
                                if tool in self.tools:
                                    recovered_args = self._get_tool_args_via_llm(
                                        question, task_id, tool
                                    )
                                    if recovered_args:
                                        decision = {
                                            "tool_name": tool,
                                            "tool_args": recovered_args,
                                        }
                                        print(f"[Agent] Recovered with tool: {tool}")
                                        break
                            if not decision:
                                print("[Agent] Could not recover appropriate tool.")
                                return None
                    elif tool_name_candidate in self.tools:
                        if "tool_args" not in decision_candidate or not isinstance(
                            decision_candidate.get("tool_args"), dict
                        ):
                            print(
                                f"[Agent] Tool {tool_name_candidate} selected but args missing/invalid. Attempting recovery..."
                            )
                            recovered_args = self._get_tool_args_via_llm(
                                question, task_id, tool_name_candidate
                            )
                            if recovered_args is not None and isinstance(
                                recovered_args, dict
                            ):
                                decision_candidate["tool_args"] = recovered_args
                                decision = decision_candidate
                                print(
                                    f"[Agent] Successfully recovered args: {decision}"
                                )
                            else:
                                print(
                                    f"[Agent] Failed to recover args for {tool_name_candidate}."
                                )
                                return None
                        else:
                            # Validate args against tool requirements
                            tool_func = self.tools[tool_name_candidate]
                            sig = inspect.signature(tool_func)
                            required_params = {
                                name: param
                                for name, param in sig.parameters.items()
                                if param.default == inspect.Parameter.empty
                            }

                            # Check if all required parameters are present
                            missing_params = [
                                param
                                for param in required_params
                                if param not in decision_candidate["tool_args"]
                            ]

                            if missing_params:
                                print(
                                    f"[Agent] Missing required parameters: {missing_params}"
                                )
                                recovered_args = self._get_tool_args_via_llm(
                                    question, task_id, tool_name_candidate
                                )
                                if recovered_args:
                                    decision_candidate["tool_args"].update(
                                        recovered_args
                                    )
                                    decision = decision_candidate
                                else:
                                    return None
                            else:
                                decision = decision_candidate
                    else:
                        print(
                            f"[Agent] LLM selected unknown tool: {tool_name_candidate}"
                        )
                        # Attempt to recover with a similar tool
                        similar_tools = [
                            tool
                            for tool in self.tools
                            if tool.lower().replace("_", "")
                            in tool_name_candidate.lower().replace("_", "")
                            or tool_name_candidate.lower().replace("_", "")
                            in tool.lower().replace("_", "")
                        ]
                        if similar_tools:
                            print(f"[Agent] Found similar tools: {similar_tools}")
                            for tool in similar_tools:
                                recovered_args = self._get_tool_args_via_llm(
                                    question, task_id, tool
                                )
                                if recovered_args:
                                    decision = {
                                        "tool_name": tool,
                                        "tool_args": recovered_args,
                                    }
                                    print(
                                        f"[Agent] Recovered with similar tool: {tool}"
                                    )
                                    break
                        if not decision:
                            return None
                else:
                    print(f"[Agent] Invalid decision format: {cleaned_response}")
                    return None

            except json.JSONDecodeError as e:
                print(
                    f"[Agent] Failed to parse tool decision JSON: {e}. Response: {llm_response}"
                )
                # Enhanced JSON extraction
                matches = re.finditer(r"\{(?:[^{}]|(?R))*\}", llm_response)
                for match in matches:
                    try:
                        extracted = json.loads(match.group(0))
                        if isinstance(extracted, dict) and "tool_name" in extracted:
                            print(
                                f"[Agent] Successfully extracted decision from partial JSON"
                            )
                            decision = extracted
                            break
                    except:
                        continue
                if not decision:
                    return None
            except Exception as e:
                print(f"[Agent] Unexpected error in tool decision: {e}")
                return None

        # Final validation and verification
        if decision and isinstance(decision, dict) and "tool_name" in decision:
            tool_name = decision["tool_name"]

            # Verify decision against reasoning
            if tool_name == "none":
                if reasoning["tools_needed"]:
                    print(
                        "[Agent] Warning: 'none' selected but tools suggested by reasoning"
                    )
                    return None
                print("[Agent] Verified 'none' decision against reasoning")
                return decision
            elif tool_name in self.tools:
                # Verify tool matches reasoning requirements
                if tool_name not in reasoning.get("tools_needed", []):
                    print(f"[Agent] Warning: {tool_name} not in suggested tools")
                    # Continue anyway if tool seems appropriate

                if "tool_args" not in decision or not isinstance(
                    decision["tool_args"], dict
                ):
                    print(f"[Agent] Tool {tool_name} missing valid args in final check")
                    return None

                # Verify args handle edge cases
                edge_cases = reasoning.get("edge_cases", [])
                if edge_cases:
                    print(
                        f"[Agent] Verifying tool args handle edge cases: {edge_cases}"
                    )
                    # Could enhance arg validation here

                print(
                    f"[Agent] Verified {tool_name} selection with args: {decision['tool_args']}"
                )
                return decision
            else:
                print(f"[Agent] Final validation failed: unknown tool {tool_name}")
                return None
        else:
            print("[Agent] Final validation failed: invalid decision structure")
            return None

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Executes the selected tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Dictionary of arguments for the tool.

        Returns:
            String result of the tool execution, or error message if failed.
        """
        try:
            if not tool_name in self.tools:
                raise KeyError(f"Tool {tool_name} not found in available tools.")

            tool_function = self.tools[tool_name]
            if not callable(tool_function):
                raise TypeError(f"Tool {tool_name} is not a callable function.")

            # Argument validation against signature
            sig = inspect.signature(tool_function)
            try:
                # Validate required arguments are present
                missing_args = [
                    param.name
                    for param in sig.parameters.values()
                    if param.default == inspect.Parameter.empty
                    and param.name not in tool_args
                ]
                if missing_args:
                    raise TypeError(
                        f"Missing required arguments for {tool_name}: {', '.join(missing_args)}"
                    )

                # Remove any extra arguments not in the signature
                valid_args = {k: v for k, v in tool_args.items() if k in sig.parameters}

                # Bind and apply defaults
                bound_args = sig.bind(**valid_args)
                bound_args.apply_defaults()

                print(
                    f"[Agent] Executing {tool_name} with validated args: {bound_args.arguments}"
                )
                result = tool_function(**bound_args.arguments)

                # Ensure result is string
                if result is None:
                    return "Tool execution completed but returned no result."
                return str(result)

            except TypeError as e:
                print(
                    f"[Agent] Tool Execution Error: Argument mismatch for tool {tool_name}. Args provided: {tool_args}. Error: {e}"
                )
                return f"Error: Tool argument mismatch for {tool_name}. Details: {e}"
            except Exception as e:
                print(
                    f"[Agent] Tool Execution Error: Unexpected error in tool {tool_name}. Error: {e}"
                )
                import traceback

                traceback.print_exc()  # Log stack trace for debugging
                return f"Error: Tool {tool_name} failed during execution. Details: {e}"

        except KeyError as e:
            print(f"[Agent] Tool Execution Error: {str(e)}")
            return f"Error: {str(e)}"
        except Exception as e:
            print(
                f"[Agent] Tool Execution Error: Unexpected error executing tool {tool_name}: {e}"
            )
            import traceback

            traceback.print_exc()
            return f"Error: Failed to execute tool {tool_name}. {e}"

    def _get_simple_llm_answer(self, question: str) -> str:
        """
        Gets a direct answer from the LLM without using tools.

        Args:
            question: The user question.

        Returns:
            LLM's direct answer to the question.
        """
        print("[Agent] Getting simple LLM answer (no tool used)...")
        prompt = (
            "You are an AI assistant. Answer the following question and provide ONLY the final answer in the specified format.\n\n"
            "Rules for FINAL ANSWER format:\n"
            "1. For numbers: No commas, no units unless requested, use plain digits\n"
            "2. For strings: No articles, no abbreviations, spell out numbers unless specified otherwise\n"
            "3. For lists: No spaces after commas, no trailing comma\n\n"
            "Your response MUST follow this exact format:\n"
            "FINAL ANSWER: [answer]\n\n"
            "Where [answer] is either:\n"
            "- A single number (e.g., '42')\n"
            "- A single word or short phrase (e.g., 'Paris')\n"
            "- A comma-separated list (e.g., '2,3,5' or 'Tokyo,Yokohama,Osaka')\n\n"
            f"Question: {question}\n\n"
            "Remember: Provide ONLY the FINAL ANSWER line with no additional text, reasoning, or explanation."
        )
        # Use low temperature for deterministic answers
        llm_answer = get_llm_response(
            prompt,
            temperature=0.1,
            max_new_tokens=100,
            system_prompt=self.system_prompt,
        )

        # Extract only the FINAL ANSWER part
        match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", llm_answer, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return llm_answer.strip()  # Fallback to full answer if format not found

    def _get_final_answer_with_tool_result(
        self, question: str, tool_result: str
    ) -> str:
        """
        Synthesizes the final answer using the question and tool result.
        Now includes RAG capabilities for enhanced accuracy.

        Args:
            question: The original user question.
            tool_result: Result from the executed tool.

        Returns:
            Synthesized answer combining the tool result with the question context.
        """
        print("[Agent] Synthesizing final answer using tool result...")

        # First, apply chain-of-thought reasoning to understand the question better
        reasoning = self._apply_chain_of_thought(question)

        # For temporal or factual questions, try to verify the information
        if reasoning["question_type"]["primary"] in [
            "temporal",
            "factual",
        ] and not tool_result.startswith("Error:"):
            try:
                # Use web search to verify the information
                verification_query = f"verify {question}"
                verification_result = (
                    self.tools["web_search"](verification_query)
                    if "web_search" in self.tools
                    else ""
                )

                prompt = (
                    "You are an AI assistant tasked with providing a highly accurate answer.\n"
                    "You have:\n"
                    "1. Primary tool result\n"
                    "2. Verification data\n"
                    "3. Question analysis\n\n"
                    "Rules for FINAL ANSWER format:\n"
                    "1. For numbers: No commas, no units unless requested, use plain digits\n"
                    "2. For strings: No articles, no abbreviations, spell out numbers unless specified otherwise\n"
                    "3. For lists: No spaces after commas, no trailing comma\n\n"
                    "Your response MUST follow this exact format:\n"
                    "FINAL ANSWER: [answer]\n\n"
                    "Where [answer] is either:\n"
                    "- A single number (e.g., '42')\n"
                    "- A single word or short phrase (e.g., 'Paris')\n"
                    "- A comma-separated list (e.g., '2,3,5' or 'Tokyo,Yokohama,Osaka')\n\n"
                    f"Question: {question}\n\n"
                    f"Question Analysis:\n{json.dumps(reasoning, indent=2)}\n\n"
                    f'Primary Tool Result:\n"""{tool_result}"""\n\n'
                    f'Verification Data:\n"""{verification_result}"""\n\n'
                    "Instructions:\n"
                    "1. Compare the primary result with verification data\n"
                    "2. Resolve any conflicts using logical reasoning\n"
                    "3. Provide the most accurate answer in the required format\n"
                    "4. If information conflicts, use the most reliable source\n"
                    "5. If verification fails, indicate uncertainty in the answer\n\n"
                    "Remember: Provide ONLY the FINAL ANSWER line with no additional text, reasoning, or explanation."
                )
            except Exception as e:
                print(f"[Agent] Verification attempt failed: {e}")
                # Fall back to basic prompt if verification fails
                prompt = self._get_basic_synthesis_prompt(question, tool_result)
        else:
            # Use basic prompt for other types of questions or error results
            prompt = self._get_basic_synthesis_prompt(question, tool_result)

        # Use low temperature for deterministic synthesis
        llm_answer = get_llm_response(
            prompt,
            temperature=0.1,
            max_new_tokens=100,
            system_prompt=self.system_prompt,
        )

        # Extract only the FINAL ANSWER part
        match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", llm_answer, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return llm_answer.strip()  # Fallback to full answer if format not found

    def _get_basic_synthesis_prompt(self, question: str, tool_result: str) -> str:
        """Helper method to get the basic synthesis prompt without RAG."""
        return (
            "You are an AI assistant. Based on the tool result, provide ONLY the final answer in the specified format.\n\n"
            "Rules for FINAL ANSWER format:\n"
            "1. For numbers: No commas, no units unless requested, use plain digits\n"
            "2. For strings: No articles, no abbreviations, spell out numbers unless specified otherwise\n"
            "3. For lists: No spaces after commas, no trailing comma\n\n"
            "Your response MUST follow this exact format:\n"
            "FINAL ANSWER: [answer]\n\n"
            "Where [answer] is either:\n"
            "- A single number (e.g., '42')\n"
            "- A single word or short phrase (e.g., 'Paris')\n"
            "- A comma-separated list (e.g., '2,3,5' or 'Tokyo,Yokohama,Osaka')\n\n"
            f"Question: {question}\n\n"
            f'Tool Result:\n"""{tool_result}"""\n\n'
            "Remember: Provide ONLY the FINAL ANSWER line with no additional text, reasoning, or explanation.\n"
            "If the tool result indicates an error or doesn't contain the necessary information, respond with 'Error: [brief reason]'"
        )

    def _clean_response(self, text: str) -> str:
        """
        Basic cleaning for the final response.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text with normalized whitespace and no code blocks.
        """
        if not isinstance(text, str):
            text = str(text)  # Ensure it's a string

        # Remove markdown code blocks
        text = re.sub(r"^```(?:json|text)?\n?|\n?```$", "", text)

        # Remove any "FINAL ANSWER:" prefix if present
        text = re.sub(r"^FINAL ANSWER:\s*", "", text, flags=re.IGNORECASE)

        # Remove any trailing/leading quotes
        text = text.strip("\"'")

        # Normalize whitespace
        text = " ".join(text.split())

        # Remove any trailing/leading punctuation except when part of a valid answer
        text = text.strip(".,;: ")

        return text

    def _validate_answer(
        self, answer: str, question: str, reasoning: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """
        Enhanced validation of the generated answer through multi-step verification.

        Args:
            answer: The generated answer
            question: The original question
            reasoning: The chain-of-thought reasoning dictionary

        Returns:
            Tuple of (is_valid, reason, confidence_score)
        """
        prompt = (
            "You are a critical validator tasked with ensuring answer quality and accuracy.\n"
            "Analyze the following answer against the question, reasoning, and validation criteria:\n\n"
            f"Question: {question}\n"
            f"Generated Answer: {answer}\n"
            f"Question Analysis: {json.dumps(reasoning, indent=2)}\n\n"
            "Validation Framework:\n\n"
            "1. Format Compliance\n"
            "   - Matches required format (number, short phrase, or comma-separated list)\n"
            "   - Follows formatting rules (no commas in numbers, no units unless requested)\n"
            "   - Contains no extraneous information\n\n"
            "2. Semantic Accuracy\n"
            "   - Directly addresses the question\n"
            "   - Matches the expected answer type\n"
            "   - Contains all necessary information\n\n"
            "3. Logical Consistency\n"
            "   - Follows from the reasoning steps\n"
            "   - Handles edge cases appropriately\n"
            "   - Maintains internal consistency\n\n"
            "4. Edge Case Handling\n"
            "   - Accounts for ambiguities\n"
            "   - Handles boundary conditions\n"
            "   - Follows specified rules for special cases\n\n"
            "**Crucially, compare the answer to the `core_facts` and `solution_steps` in the Question Analysis. Heavily penalize the `semantic_score` if the answer ignores key constraints or components of the question.**\n\n"
            "Respond with a JSON object:\n"
            "{\n"
            '  "is_valid": true/false,\n'
            '  "validation_details": {\n'
            '    "format_score": 0-100,\n'
            '    "semantic_score": 0-100,\n'
            '    "logical_score": 0-100,\n'
            '    "edge_case_score": 0-100\n'
            "  },\n"
            '  "issues": [\n'
            '    {"category": "format|semantic|logical|edge_case", "description": "issue description", "severity": "high|medium|low"}\n'
            "  ],\n"
            '  "confidence_score": 0-100,\n'
            '  "improvement_suggestions": ["suggestion1", "suggestion2"]\n'
            "}"
        )

        try:
            response = get_llm_response(
                prompt,
                temperature=0.1,
                max_new_tokens=500,
                system_prompt=self.system_prompt,
            )
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                validation = json.loads(match.group(0))

                # Calculate weighted validation score
                scores = validation.get("validation_details", {})
                weighted_score = (
                    scores.get("format_score", 0) * 0.3
                    + scores.get("semantic_score", 0) * 0.3
                    + scores.get("logical_score", 0) * 0.25
                    + scores.get("edge_case_score", 0) * 0.15
                )

                # Get critical issues
                critical_issues = [
                    issue["description"]
                    for issue in validation.get("issues", [])
                    if issue.get("severity") == "high"
                ]

                # Determine validity and construct reason
                is_valid = validation.get("is_valid", False) and weighted_score >= 85
                reason = (
                    "Critical issues found: " + "; ".join(critical_issues)
                    if critical_issues
                    else validation.get(
                        "improvement_suggestions", ["No specific issues"]
                    )[0]
                )

                confidence = validation.get("confidence_score", 0)
                return is_valid, reason, confidence

            return False, "Validation parsing failed", 0.0
        except Exception as e:
            print(f"[Agent] Enhanced validation error: {e}")
            return False, f"Validation error: {str(e)}", 0.0

    def _refine_answer(
        self,
        answer: str,
        question: str,
        reasoning: Dict[str, Any],
        validation_feedback: str,
        confidence: float,
    ) -> str:
        """
        Enhanced answer refinement with multi-strategy improvement.

        Args:
            answer: The original answer
            question: The original question
            reasoning: The chain-of-thought reasoning
            validation_feedback: Why the original answer was invalid
            confidence: Confidence score from validation

        Returns:
            Refined answer string
        """
        prompt = (
            "You are an AI assistant tasked with refining an answer that failed validation.\n\n"
            f"Original Question: {question}\n"
            f"Original Answer: {answer}\n"
            f"Validation Feedback: {validation_feedback}\n"
            f"Confidence Score: {confidence}\n"
            f"Question Analysis: {json.dumps(reasoning, indent=2)}\n\n"
            "Refinement Framework:\n\n"
            "1. Answer Format Requirements\n"
            "   - For numbers: No commas, no units unless requested, use plain digits\n"
            "   - For strings: No articles, no abbreviations, spell out numbers unless specified\n"
            "   - For lists: No spaces after commas, no trailing comma\n\n"
            "2. Improvement Strategies\n"
            "   - Address all validation issues\n"
            "   - Ensure complete coverage of question requirements\n"
            "   - Handle edge cases appropriately\n"
            "   - Maintain consistency with reasoning steps\n\n"
            "3. Quality Checks\n"
            "   - Verify format compliance\n"
            "   - Check logical consistency\n"
            "   - Validate against edge cases\n"
            "   - Ensure answer completeness\n\n"
            "Instructions:\n"
            "1. Analyze the validation feedback\n"
            "2. Apply appropriate refinement strategies\n"
            "3. Verify the refined answer against all requirements\n"
            "4. Ensure the answer is in the exact required format\n\n"
            "Provide ONLY the refined answer in the format:\n"
            "FINAL ANSWER: [refined_answer]"
        )

        # Use lower temperature for refinement to ensure consistency
        refined = get_llm_response(
            prompt,
            temperature=0.05,
            max_new_tokens=100,
            system_prompt=self.system_prompt,
        )
        match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", refined, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return refined.strip()

    def run(
        self,
        question: str,
        task_id: Optional[str] = None,
        force_tool_name: Optional[str] = None,
        max_refinement_attempts: int = 3,
        min_confidence_threshold: float = 85.0,
        max_tool_attempts: int = 3,
    ) -> str:
        """
        Enhanced agent logic with iterative refinement and confidence thresholds.

        Args:
            question: The user's question.
            task_id: Optional GAIA task ID.
            force_tool_name: Optional tool name to force use of.
            max_refinement_attempts: Maximum number of refinement attempts.
            min_confidence_threshold: Minimum confidence score to accept an answer.
            max_tool_attempts: Maximum number of attempts to find and use a working tool.

        Returns:
            Final answer string, or error message if something went wrong.
        """
        print(f"\n[Agent] === Starting Task ID: {task_id} ===")
        print(f"[Agent] Received Question: {question}")

        try:
            # Apply initial chain-of-thought reasoning
            reasoning = self._apply_chain_of_thought(question)

            # Get initial answer through normal process, with fallback logic
            initial_answer = self._get_initial_answer(
                question,
                task_id,
                force_tool_name,
                max_tool_attempts=max_tool_attempts,
            )

            # Iterative refinement loop with confidence tracking
            current_answer = initial_answer
            best_answer = initial_answer
            best_confidence = 0.0

            for attempt in range(max_refinement_attempts):
                # Validate the current answer
                is_valid, validation_reason, confidence = self._validate_answer(
                    current_answer, question, reasoning
                )

                # Track best answer based on confidence
                if confidence > best_confidence:
                    best_answer = current_answer
                    best_confidence = confidence

                # Check if we've reached our quality threshold
                if is_valid and confidence >= min_confidence_threshold:
                    print(
                        f"[Agent] Answer validated successfully on attempt {attempt + 1} "
                        f"with confidence {confidence:.2f}%"
                    )
                    return self._clean_response(current_answer)

                print(
                    f"[Agent] Refining answer. Attempt {attempt + 1}. "
                    f"Reason: {validation_reason}, Confidence: {confidence:.2f}%"
                )

                # Apply refinement
                current_answer = self._refine_answer(
                    current_answer,
                    question,
                    reasoning,
                    validation_reason,
                    confidence,
                )

            # If we've exhausted attempts, return the best answer we found
            print(
                f"[Agent] Max refinement attempts reached. Using best answer "
                f"(confidence: {best_confidence:.2f}%)"
            )
            return self._clean_response(best_answer)

        except Exception as e:
            print(
                f"[Agent] ***** A critical error occurred during agent run: {e} *****"
            )
            import traceback

            traceback.print_exc()
            return f"Error: Agent encountered an unexpected critical error during execution. Check logs for details. Error type: {type(e).__name__}"

    def _get_initial_answer(
        self,
        question: str,
        task_id: Optional[str],
        force_tool_name: Optional[str],
        max_tool_attempts: int,
    ) -> str:
        """Helper method to get the initial answer using tool logic with retries."""
        failed_tools = set()
        for attempt in range(max_tool_attempts):
            print(
                f"[Agent] Tool selection attempt {attempt + 1}/{max_tool_attempts}..."
            )
            # 1. Decide if a tool is needed (or use forced tool)
            tool_decision = self._get_tool_decision(
                question, task_id, force_tool_name, failed_tools
            )

            # 2. Execute tool if decided, otherwise get simple answer
            if tool_decision and tool_decision.get("tool_name") != "none":
                tool_name = tool_decision["tool_name"]
                tool_args = tool_decision.get("tool_args", {})

                # Handle file analysis cases
                if tool_name == "analyze_file_content" and "file_path" not in tool_args:
                    if task_id:
                        print(
                            "[Agent] `analyze_file_content` chosen, but `file_path` missing. Attempting implicit download..."
                        )
                        download_result_path = self._execute_tool(
                            "download_gaia_file", {"task_id": task_id}
                        )
                        if isinstance(
                            download_result_path, str
                        ) and not download_result_path.startswith("Error:"):
                            tool_args["file_path"] = download_result_path
                        else:
                            # Treat download failure as a tool failure
                            print(
                                f"[Agent] Implicit download failed. Adding 'download_gaia_file' to failed tools."
                            )
                            failed_tools.add("download_gaia_file")
                            continue  # Retry with a different tool
                    else:
                        return self._clean_response(
                            "Error: File analysis requested but no task_id provided"
                        )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)

                # Handle tool execution results
                if isinstance(tool_result, str) and tool_result.startswith("Error:"):
                    print(
                        f"[Agent] Tool '{tool_name}' failed. Adding to failed list and retrying."
                    )
                    failed_tools.add(tool_name)
                    # For the last attempt, return the error
                    if attempt == max_tool_attempts - 1:
                        return self._get_final_answer_with_tool_result(
                            question, tool_result
                        )
                    continue  # Go to the next attempt
                else:
                    # Tool succeeded, synthesize answer and return
                    return self._get_final_answer_with_tool_result(
                        question, str(tool_result)
                    )
            elif tool_decision and tool_decision.get("tool_name") == "none":
                # If LLM decides no tool is needed, respect that and exit the loop
                return self._get_simple_llm_answer(question)
            else:
                # If no decision could be made, try again
                print("[Agent] Could not make a tool decision. Retrying...")
                continue

        # If loop finishes without success, return a message
        print("[Agent] All tool attempts failed. Falling back to simple answer.")
        return self._get_simple_llm_answer(question)

    def __call__(
        self,
        question: str,
        task_id: Optional[str] = None,
        force_tool_name: Optional[str] = None,
    ) -> str:
        """
        Makes the agent callable directly.

        Args:
            question: The user's question.
            task_id: Optional GAIA task ID.
            force_tool_name: Optional tool name to force use of.

        Returns:
            Final answer string, or error message if something went wrong.
        """
        return self.run(question, task_id, force_tool_name)


# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing GaiaAgent with Groq LLM Client ---")
    # Ensure required libraries are installed and HF_TOKEN is set if needed.

    # Test requires tools.py and llm_client_hf.py in the same directory
    # and necessary dependencies installed.

    try:
        print("Initializing agent (this will load the LLM)...")
        agent = GaiaAgent()
        print("Agent initialized.")

        # Test case 1: Simple question (should ideally use 'none' tool)
        test_question_1 = "What is the capital of France?"
        print(f"\n>>> Testing with: {test_question_1}")
        answer_1 = agent.run(test_question_1, task_id="test_simple")
        print(f"Agent Answer 1: {answer_1}")

        # Test case 2: Question likely needing web search
        # Note: This requires the web_search tool in tools.py to be functional.
        test_question_2 = (
            "What were the major AI advancements announced by Google in the last month?"
        )
        print(f"\n>>> Testing with: {test_question_2}")
        answer_2 = agent.run(test_question_2, task_id="test_websearch")
        print(f"Agent Answer 2: {answer_2}")

        # Test case 3: Question needing math tool
        # Note: This requires the math_tool in tools.py to be functional.
        test_question_3 = "Calculate the result of 25 * (1 + 15%) "
        print(f"\n>>> Testing with: {test_question_3}")
        answer_3 = agent.run(test_question_3, task_id="test_math")
        print(f"Agent Answer 3: {answer_3}")

        # Test case 4: Question potentially needing file download (requires task_id)
        # Note: This requires download_gaia_file and potentially analyze_file_content tools.
        # The GAIA API isn't running here, so download will likely fail, but tests the logic.
        test_question_4 = (
            "Analyze the sentiment of the document associated with this task."
        )
        print(f"\n>>> Testing with: {test_question_4} (Task ID: gaia-test-doc)")
        answer_4 = agent.run(test_question_4, task_id="gaia-test-doc")
        print(f"Agent Answer 4: {answer_4}")

    except Exception as e:
        print(f"\n--- Agent Test Failed --- ")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Agent Test Complete ---")
