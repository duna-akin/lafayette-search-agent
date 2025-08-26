# 🎓 Lafayette College Information Assistant

An AI-powered chatbot that provides comprehensive information about Lafayette College by searching and analyzing content from the official Lafayette.edu website.

## Features

- **Lafayette-Specific**: Only provides information about Lafayette College
- **Real-time Search**: Searches the official lafayette.edu website for up-to-date information
- **Comprehensive Coverage**: Answers questions about admissions, academics, campus life, financial aid, and more
- **Interactive Chat**: User-friendly Streamlit interface with conversation history
- **Intelligent Responses**: Uses OpenAI's GPT-4 to provide well-structured, helpful answers

## What You Can Ask About

### 📋 Admissions

- Application deadlines (Early Decision I, Early Decision II, Regular Decision)
- Application requirements and process
- Admission statistics and requirements
- Campus visits and interviews

### 🎓 Academics

- Available majors and programs
- Course offerings and curriculum
- Faculty information
- Research opportunities
- Study abroad programs

### 🏫 Campus Life

- Housing and dining options
- Student organizations and clubs
- Campus events and traditions
- Recreation and fitness facilities

### 💰 Financial Aid

- Scholarships and grants
- Work-study opportunities
- Financial aid application process
- Cost of attendance

### 🏃‍♂️ Athletics

- Sports teams and divisions
- Athletic facilities
- Intramural sports
- Recreation programs

## Installation

1. Clone this repository:

```bash
git clone <your-repo-url>
cd search_engine_agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit application:

```bash
streamlit run main.py
```

2. Open your browser to the provided URL (usually `http://localhost:8501`)

3. Start asking questions about Lafayette College!

## Sample Questions

- "What are the application deadlines for Lafayette College?"
- "What majors does Lafayette offer in engineering?"
- "How do I apply for financial aid at Lafayette?"
- "What is campus life like at Lafayette College?"
- "What research opportunities are available for undergraduates?"
- "Tell me about Lafayette's study abroad programs"

## Technical Details

- **Framework**: Streamlit for the web interface
- **AI Model**: OpenAI GPT-4o-mini for intelligent responses
- **Search Method**: Web scraping of lafayette.edu using requests and BeautifulSoup
- **State Management**: LangGraph for conversation flow
- **Search Strategy**: Site-specific Google search + direct content scraping

## Limitations

- Only provides information about Lafayette College
- Dependent on the availability and structure of lafayette.edu
- Information accuracy depends on the content available on the official website
- Rate-limited to be respectful to the Lafayette College website

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes. Please respect Lafayette College's website terms of service when using this tool.
