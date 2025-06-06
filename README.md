# 🤖 GAIA Agent - Advanced AI Problem Solver

**Version 1.0** | **GAIA Benchmark - Level 1 Questions**

Welcome to my final hands-on project for the [Hugging Face Agent Course](https://huggingface.co/learn/agents-course/en/unit0/introduction)!

In this project, I designed my first AI agent, capable of solving complex, multi-step problems using advanced reasoning, tool orchestration, and natural language understanding. 

The agent is specifically built to be evaluated on a subset of **Level 1** questions from the **GAIA benchmark**, a framework designed to test and measure AI agents' reasoning and tool-using capabilities.

The entire system is wrapped in a Gradio-based web interface, allowing users to run full evaluations and submit answers to a scoring service.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-red.svg)](https://gradio.app)
[![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)](https://groq.com)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)

---

## 🎯 **Key Features**

### 🧠 **Advanced Reasoning Engine**
- **🔥 Chain-of-Thought Analysis**: Breaks down complex problems systematically
- **⚡ Dynamic Tool Selection**: Intelligently chooses appropriate tools for each task
- **🔄 Iterative Refinement**: Validates and improves answers until confidence threshold is met
- **📊 Multi-Domain Problem Solving**: Handles web search, data analysis, image processing, and more

### 🛠️ **Comprehensive Tool Suite**
- **🌐 Web & Research**: Web search, Wikipedia lookup, webpage scraping
- **📊 Data Analysis**: CSV/Excel processing with Pandas integration
- **🖼️ Multimedia Processing**: Image analysis with OpenCV, YouTube transcript extraction
- **🧮 Mathematical Operations**: Equation solving, string manipulation, chess analysis
- **📁 File Operations**: Intelligent file download and content analysis

### 🔒 **Enterprise-Grade Architecture**
- **Modular Design**: Clean separation between UI, agent logic, and tools
- **Error Handling**: Failure recovery with retry mechanisms
- **API Integration**: Seamless connection to GAIA evaluation service
- **Standardized Evaluation**: Built-in benchmark testing and scoring

### 🌐 **Modern Web Interface**
- **Gradio-Based UI**: Intuitive, responsive web interface
- **Real-Time Feedback**: Live progress updates during evaluation
- **Detailed Logging**: Complete audit trail of reasoning and results
- **HuggingFace Integration**: Direct authentication and deployment support

---

## 🚀 **Quick Start**

```bash
# 1. Clone the repository
git clone <repository-url>
cd gaia-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Launch the application
python app.py
```

**Try it out:**
- Open the web interface at `http://localhost:7860`
- Click "Run Evaluation" to test against GAIA benchmark
- Watch the agent solve complex problems in real-time!

---

## 🏗️ **System Architecture**

### 📋 **Processing Pipeline**
```
User → Gradio UI → GaiaEvaluationRunner → GaiaAgent → Tools → LLM → Scoring Service → Results

```

### 🔧 **Core Components**

| Component | File | Responsibility |
|-----------|------|----------------|
| **🎨 User Interface** | `app.py` | Gradio web app, evaluation orchestration |
| **🧠 Agent Core** | `agent.py` | Chain-of-thought reasoning, tool coordination |
| **🛠️ Tool Library** | `tools.py` | Specialized function implementations |
| **🤖 LLM Client** | `llm_client.py` | Groq API communication and LLM model selection layer |
| **📝 System Prompt** | `gaia_system_prompt.py` | Agent persona and instructions |

---


## 💡 **How It Works**

### 🎯 **The GAIA Agent Problem-Solving Process**

1. **📥 Question Analysis**
   - Receives complex, multi-step questions from GAIA benchmark
   - Performs chain-of-thought reasoning to understand requirements
   - Identifies key information needed and potential solution paths

2. **🔧 Tool Selection & Orchestration**
   ```python
   # Dynamic tool selection based on question context
   if question_needs_web_search():
       result = tools.web_search(query)
   elif question_needs_data_analysis():
       result = tools.analyze_data(file_path)
   elif question_needs_image_processing():
       result = tools.analyze_image(image_path)
   ```

3. **⚡ Execution & Synthesis**
   - Executes selected tools with generated parameters
   - Combines tool outputs with original question context
   - Synthesizes coherent, accurate final answers

4. **🔍 Validation & Refinement**
   ```python
   while confidence < threshold and attempts < max_attempts:
       answer = validate_and_refine(current_answer)
       confidence = assess_confidence(answer)
       attempts += 1
   ```

### 🧰 **Available Tools Showcase**

#### 🌐 **Research & Information Gathering**
```python
web_search(query)           # Tavily-powered web search
wiki_search(topic)          # Wikipedia knowledge lookup  
scrape_webpage(url)         # Content extraction from URLs
```

#### 📊 **Data Analysis & Processing**
```python
analyze_data(file_path)     # CSV/Excel analysis with Pandas
download_gaia_file(url)     # Intelligent file downloading
analyze_file_content(path)  # Multi-format content analysis
```

#### 🎥 **Multimedia & Advanced Processing**
```python
get_youtube_transcript(url) # Video transcript extraction
analyze_image(image_path)   # OpenCV-powered image analysis
analyze_chess_position(fen) # Chess position evaluation
```

#### 🧮 **Mathematical & Logical Operations**
```python
solve_equation(expression)  # Mathematical problem solving
string_operation(text, op)  # Text manipulation utilities
```

---

## 📦 **Installation & Setup**

### 🔧 **System Requirements**
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for API services
- Modern web browser for UI access

### 📥 **Detailed Installation**

#### **Method 1: Standard Installation**
```bash
# Clone repository
git clone https://github.com/your-username/gaia-agent.git
cd gaia-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### **Method 2: Docker Installation**
```bash
# Build Docker image
docker build -t gaia-agent .

# Run container
docker run -p 7860:7860 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e TAVILY_API_KEY=$TAVILY_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  gaia-agent
```

### 🔑 **Environment Configuration**

Create a `.env` file with your API credentials:

```bash
# === REQUIRED API KEYS ===
GROQ_API_KEY=gsk_your_groq_api_key_here          # Groq LLM API
TAVILY_API_KEY=tvly-your_tavily_key_here         # Web search API
HF_TOKEN=hf_your_huggingface_token_here          # HuggingFace 

```

### **Run GAIA Evaluation**

1. **Authentication**: Ensure your HuggingFace token is configured
2. **Start Evaluation**: Click "Run Evaluation" button in the web interface
3. **Monitor Progress**: Watch real-time updates as the agent processes questions
4. **View Results**: Get detailed scoring and performance metrics



### **Example Problem-Solving Session**

**Question**: *"What is the population of the capital city of the country where the 2024 Olympics were held?"*

**Agent Reasoning Process**:
```
🧠 Chain-of-Thought Analysis:
1. Need to identify where 2024 Olympics were held
2. Find the capital city of that country  
3. Look up the current population of that capital

🔧 Tool Execution:
Step 1: web_search("2024 Olympics location host city")
→ Result: Paris, France hosted 2024 Summer Olympics

Step 2: web_search("capital city of France") 
→ Result: Paris is the capital of France

Step 3: web_search("Paris France current population 2024")
→ Result: Approximately 2.16 million (city proper)

✅ Final Answer: The population of Paris, the capital city of France where the 2024 Olympics were held, is approximately 2.16 million people.
```

---

## ⚙️ **Advanced Configuration**

### 🔧 **Customizing Agent Behavior**

#### **Modify System Prompt** (`gaia_system_prompt.py`):
```python
# Adjust reasoning approach
SYSTEM_PROMPT = """
You are GAIA Agent, an advanced AI problem solver...
[Customize persona, instructions, and response format]
"""
```

#### **Configure Tool Parameters** (`tools.py`):
```python
# Adjust search parameters
def web_search(query, max_results=5, include_raw_content=True):
    # Customize search behavior
    
# Add new custom tools
def your_custom_tool(parameters):
    """Your specialized functionality"""
    return result
```

#### **Tune Agent Settings** (`agent.py`):
```python
class GaiaAgent:
    def __init__(self):
        self.max_attempts = 3           # Retry limit
        self.confidence_threshold = 0.8  # Answer quality bar
        self.timeout = 300              # Tool execution limit
```

### 📊 **Performance Monitoring**

#### **Built-in Metrics Tracking**:
```python
# Automatic performance logging
metrics = {
    'response_time': 45.3,
    'tool_usage': {'web_search': 15, 'data_analysis': 8},
    'success_rate': 0.87,
    'confidence_scores': [0.9, 0.8, 0.95, ...]
}
```



---

## 🏢 **Deployment Options**

### 🚀 **HuggingFace Spaces (Recommended)**

```yaml
# spaces_config.yml
title: GAIA Agent - AI Problem Solver
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
```

**Deploy Steps**:
1. Push your code to HuggingFace repository
2. Configure secrets in Space settings
3. Your agent will be available at `https://huggingface.co/spaces/username/gaia-agent`

### ☁️ **Cloud Deployment**

#### **Google Cloud Run**:
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "app.py", "--server-name", "0.0.0.0"]
```

#### **AWS EC2 with Docker**:
```bash
# Launch EC2 instance
aws ec2 run-instances --image-id ami-0abcdef1234567890 --instance-type t3.medium

# Deploy container
docker run -d -p 80:7860 --name gaia-agent \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  your-dockerhub-username/gaia-agent
```

### 🔧 **Local Development Server**
```bash
# Development mode with auto-reload
python app.py --reload --debug

# Production mode  
gunicorn -w 4 -b 0.0.0.0:7860 app:app

```

## 🧪 **Development & Testing**

### 🔬 **Project Structure**
```
Gaia_Agent_HF_Course/
├── ⚙️ .env.example              # Environment variables template
├── 📱  app.py                    # Gradio UI & evaluation runner
├── 🧠 agent.py                  # Core agent logic & reasoning
├── 🛠️ tools.py                  # Tool implementations
├── 🤖 llm_client.py             # Groq API communication and LLM model selection
├── 📝 gaia_system_prompt.py     # Agent robust system prompt
├── 📋 requirements.txt          # Python dependencies
├── ⚙️ DIAGRAM_ARCHITECTURE.md   # Diagram architecture of the project
└── 📖 README.md                 # This documentation

```

### 🔧 **Contributing**

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch: `git checkout -b feature/amazing-improvement`
4. **Implement** your changes with tests
5. **Commit** with clear messages: `git commit -m "feat: add support for new tool type"`
6. **Push** to your fork: `git push origin feature/amazing-improvement`
7. **Create** a Pull Request with detailed description

**Development Guidelines**:
- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include docstrings for public methods
- Write tests for new functionality
- Update documentation as needed

---


## 🎖️ **Acknowledgments & References**

### 🏆 **Built With**
- **[LangChain](https://langchain.com)** - Framework for connecting LLMs with tools and data sources
- **[Groq](https://groq.com)** - Ultra-fast LLM inference with Llama and Mixtral models
- **[Gradio](https://gradio.app)** - User-friendly web interface framework
- **[Tavily](https://tavily.com)** - Advanced web search API for AI agents
- **[HuggingFace](https://huggingface.co)** - ML platform and GAIA benchmark hosting
- **[OpenCV](https://opencv.org)** - Computer vision and image processing
- **[Pandas](https://pandas.pydata.org)** - Data manipulation and analysis

### 📚 **Research & Benchmarks**
- **[GAIA Benchmark](https://arxiv.org/abs/2311.12983)**: [Mialon et al., 2023] - "GAIA: a benchmark for General AI Assistants"
- **Tool-using AI**: Research on autonomous agent architectures and tool orchestration
- **Chain-of-Thought**: Advanced reasoning techniques for large language models

### 🙏 **Special Thanks**
- **HuggingFace Team** for the comprehensive AI course and evaluation framework
- **GAIA Benchmark Creators** for establishing rigorous AI agent evaluation standards  
- **Open Source Community** for the excellent tools and libraries that made this project possible

---


## 🎉 **Conclusion**

This **GAIA Agent** repository marks the creation of my first AI Agent, developed as the final project for the Hugging Face Agents Course. While its accuracy may not yet reflect state-of-the-art performance, this project served as a valuable introduction to a technology I am deeply passionate about and plan to continue developing.

The agent demonstrates capabilities such as:

✨ **Advanced Reasoning** with chain-of-thought problem decomposition  
🛠️ **Versatile Tool Integration** across multiple domains and data types  
🎯 **Rigorous Evaluation** against standardized benchmarks  
🚀 **Production-Ready Architecture** with modern deployment options  

**Ready to solve problems that require human-level reasoning and tool use!**

---

<div align="center">

**⭐ If this project helps your research or work, please consider giving it a star!**


*Developed with ❤️ using Python, AI, and a generous dose of coffee ☕*


</div>
