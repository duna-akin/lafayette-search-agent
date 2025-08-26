import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
import getpass
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
import urllib.parse
import time

load_dotenv()

def search_lafayette_edu(query):
    """Search Lafayette College website for relevant information."""
    try:
        # Define relevant Lafayette URLs by topic
        lafayette_url_map = {
            'admissions': [
                "https://admissions.lafayette.edu/",
                "https://admissions.lafayette.edu/what-we-look-for/",
                "https://admissions.lafayette.edu/apply/",
                "https://admissions.lafayette.edu/deadlines-and-forms/",
                "https://admissions.lafayette.edu/admissions-visits/"
            ],
            'financial': [
                "https://admissions.lafayette.edu/financial-aid/",
                "https://admissions.lafayette.edu/college-costs/"
            ],
            'academics': [
                "https://academics.lafayette.edu/",
                "https://academics.lafayette.edu/departments-programs/"
            ],
            'campus': [
                "https://campuslife.lafayette.edu/"
            ],
            'about': [
                "https://about.lafayette.edu/",
                "https://about.lafayette.edu/lafayette-at-a-glance/",
                "https://about.lafayette.edu/mission-and-history/",
                "https://about.lafayette.edu/why-not/"
            ],
            'president': [
                "https://president.lafayette.edu/"
            ]
        }
        
        # Determine which URLs to prioritize based on query keywords
        query_lower = query.lower()
        relevant_urls = []
        
        if any(word in query_lower for word in ['admission', 'apply', 'application', 'deadline', 'requirement', 'ed', 'early decision', 'regular decision']):
            relevant_urls.extend(lafayette_url_map['admissions'])
        if any(word in query_lower for word in ['financial', 'aid', 'scholarship', 'cost', 'tuition', 'money', 'grant']):
            relevant_urls.extend(lafayette_url_map['financial'])
        if any(word in query_lower for word in ['academic', 'major', 'program', 'course', 'department', 'study', 'curriculum']):
            relevant_urls.extend(lafayette_url_map['academics'])
        if any(word in query_lower for word in ['campus', 'life', 'student', 'housing', 'dining', 'club', 'organization']):
            relevant_urls.extend(lafayette_url_map['campus'])
        if any(word in query_lower for word in ['about', 'history', 'mission', 'overview', 'glance', 'why']):
            relevant_urls.extend(lafayette_url_map['about'])
        if any(word in query_lower for word in ['president']):
            relevant_urls.extend(lafayette_url_map['president'])
        
        # Use Google to search specifically within lafayette.edu
        search_query = f"site:lafayette.edu {query}"
        google_search_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get search results from Google
        lafayette_urls = []
        try:
            response = requests.get(google_search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract Lafayette.edu URLs from search results
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'lafayette.edu' in href and href.startswith('http'):
                    if href not in lafayette_urls:
                        lafayette_urls.append(href)
                    if len(lafayette_urls) >= 2:  # Limit to top 2 Google results
                        break
        except:
            pass
        
        # Add relevant URLs based on query keywords (if we don't have enough from Google)
        if len(lafayette_urls) < 3:
            for url in relevant_urls:
                if url not in lafayette_urls:
                    lafayette_urls.append(url)
                if len(lafayette_urls) >= 3:
                    break
        
        # If still no URLs, use general fallback pages
        if not lafayette_urls:
            lafayette_urls = [
                "https://admissions.lafayette.edu/",
                "https://academics.lafayette.edu/",
                "https://about.lafayette.edu/"
            ]
        
        # Scrape content from Lafayette URLs
        all_content = []
        for url in lafayette_urls:
            try:
                time.sleep(1)  # Be respectful with requests
                page_response = requests.get(url, headers=headers, timeout=10)
                page_soup = BeautifulSoup(page_response.content, 'html.parser')
                
                # Extract text content
                for script in page_soup(["script", "style"]):
                    script.decompose()
                text = page_soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit content length
                if len(text) > 1000:
                    text = text[:1000] + "..."
                
                all_content.append(f"From {url}:\n{text}\n")
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        return "\n\n".join(all_content) if all_content else "No relevant information found on Lafayette College website."
        
    except Exception as e:
        return f"Error searching Lafayette.edu: {str(e)}"

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
llm = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)



def chatbot(state: State):
    """Lafayette College Information Assistant - provides comprehensive responses about Lafayette College."""
    
    # Extract the user's question from the last message
    last_message = state["messages"][-1]
    
    # Handle both dict and message object formats
    if hasattr(last_message, 'content'):
        user_question = last_message.content  # Extract the content string
    else:
        user_question = last_message["content"]  # Extract from dict format
    
    # Search Lafayette College website for relevant information
    search_result = search_lafayette_edu(user_question)
    print(f"Lafayette Search Result: {search_result}")
    
    # Lafayette-specific system prompt
    SYSTEM_PROMPT = f"""You are the Official Lafayette College Information Assistant. You are an expert on all aspects of Lafayette College and can only provide information about Lafayette College.

LAFAYETTE COLLEGE INFORMATION FROM WEBSITE: {search_result}

USER QUESTION: {user_question}

INSTRUCTIONS:
- You ONLY provide information about Lafayette College
- If the question is not related to Lafayette College, politely redirect the user to ask Lafayette-related questions
- Use the search results from the Lafayette College website to provide accurate, up-to-date information
- Be helpful and comprehensive in your responses about Lafayette College
- Include specific details like deadlines, requirements, programs, etc. when available
- If you don't have enough information from the search results, acknowledge this and suggest they visit lafayette.edu or contact admissions

LAFAYETTE COLLEGE AREAS YOU CAN HELP WITH:
- Admissions (application deadlines, requirements, Early Decision I & II, Regular Decision)
- Academic programs and majors
- Campus life and student services
- Financial aid and scholarships
- Faculty and research opportunities
- Campus facilities and resources
- Alumni information
- Athletics and extracurricular activities

Please provide a helpful, well-structured response based on the Lafayette College information above."""
    
    # Get formatted response from the LLM
    formatted_response = llm.invoke([HumanMessage(content=SYSTEM_PROMPT)])
    
    # Create a response message with the formatted content
    response_message = AIMessage(content=formatted_response.content)
    return {"messages": state["messages"] + [response_message]}

# Build the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# Streamlit UI
st.set_page_config(
    page_title="Lafayette College Information Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Lafayette College Information Assistant")
st.markdown("### Your AI guide to everything Lafayette College!")
st.markdown("Ask me about admissions, academics, campus life, and more at Lafayette College.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know about Lafayette College?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching Lafayette College website..."):
            try:
                # Use the graph to get a response
                result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
                response = result["messages"][-1].content
                
                # Display assistant response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while searching Lafayette College information: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add a sidebar with information
with st.sidebar:
    st.header("🎓 About Lafayette College Assistant")
    st.markdown("""
    This AI assistant provides information exclusively about **Lafayette College** by searching the official Lafayette.edu website.
    
    **I can help you with:**
    - 📋 **Admissions**: Application deadlines, requirements, ED I/II, RD
    - 🎓 **Academics**: Majors, programs, courses, faculty
    - 🏫 **Campus Life**: Housing, dining, activities, clubs
    - 💰 **Financial Aid**: Scholarships, grants, work-study
    - 🏃‍♂️ **Athletics**: Sports teams, facilities, recreation
    - 🔬 **Research**: Opportunities, labs, projects
    - 📍 **Campus**: Buildings, facilities, resources
    
    **Sample Questions:**
    - "What are the application deadlines for Lafayette College?"
    - "What majors does Lafayette offer?"
    - "How do I apply for financial aid?"
    - "What is campus life like at Lafayette?"
    
    **Note**: I only provide information about Lafayette College. For other topics, please visit the official website.
    """)
    
    st.divider()
    
    st.markdown("**Quick Links:**")
    st.markdown("- [Lafayette College](https://about.lafayette.edu/)")
    st.markdown("- [Admissions](https://admissions.lafayette.edu/)")
    st.markdown("- [Academics](https://academics.lafayette.edu/)")
    st.markdown("- [Campus Life](https://campuslife.lafayette.edu/)")
    st.markdown("- [Financial Aid](https://admissions.lafayette.edu/financial-aid/)")
    st.markdown("- [Application Deadlines](https://admissions.lafayette.edu/deadlines-and-forms/)")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()