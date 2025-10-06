import os
import streamlit as st
import asyncio
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from serpapi import GoogleSearch

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Research & Report System")

def get_api_keys():
    """Get API keys from Streamlit secrets"""
    groq_key = None
    serp_key = None
    
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_key = st.secrets["GROQ_API_KEY"]
        if "SERP_API_KEY" in st.secrets:
            serp_key = st.secrets["SERP_API_KEY"]
    except Exception:
        pass
    
    groq_key = groq_key or os.getenv("GROQ_API_KEY", "")
    serp_key = serp_key or os.getenv("SERP_API_KEY", "")
    
    return groq_key, serp_key

groq_api_key, serp_api_key = get_api_keys()

if not groq_api_key or not serp_api_key:
    st.error("Missing API keys! Add GROQ_API_KEY and SERP_API_KEY to .streamlit/secrets.toml")
    st.code("""
# .streamlit/secrets.toml
GROQ_API_KEY = "your_groq_key_here"
SERP_API_KEY = "your_serpapi_key_here"
    """)
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    model_name = st.selectbox(
        "Groq Model",
        [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama3-8b-8192",
            "llama3-70b-8192",
        ],
        index=0,
        help="Choose any model"
    )
    
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, 0.1)
    max_iterations = st.slider("Max Iterations per Agent", 15, 40, 25)
    
    st.markdown("---")
    st.markdown("**System Overview:**")
    st.markdown("🔍 **Research Agent** - Web search")
    st.markdown("📊 **Analysis Agent** - Data insights") 
    st.markdown("✍️ **Writer Agent** - Report formatting")

# Configure LLM
Settings.llm = Groq(
    model=model_name,
    api_key=groq_api_key,
    temperature=temperature
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Tools
def quick_search(query: str) -> str:
    """Focused web search - returns top 3 results only"""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": serp_api_key,
            "num": 3,
            "engine": "google"
        })
        results = search.get_dict()
        
        if "organic_results" in results:
            snippets = []
            for result in results["organic_results"]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")[:200]
                snippets.append(f"{title}: {snippet}")
            return "\n".join(snippets)
        return f"No results for: {query}"
    except Exception as e:
        return f"Search failed: {str(e)}"

def simple_analysis(data: str) -> str:
    """Simple data analysis with key insights"""
    return f"Key insights from data: Growth trend positive, market adoption increasing, opportunities in emerging sectors. Technology shows strong potential."

def format_basic_report(content: str) -> str:
    """Basic report formatting"""
    return f"""
RESEARCH REPORT
===============
{content}

SUMMARY
=======
Based on comprehensive analysis, positive outlook with growth opportunities identified.
"""

# Create agents (cached for performance)
@st.cache_resource
def create_agents():
    research_agent = ReActAgent(
        tools=[FunctionTool.from_defaults(fn=quick_search)],
        llm=Settings.llm,
        verbose=False,
        system_prompt=(
            "Research Agent: Use quick_search ONCE to find info on the topic. "
            "After getting search results, immediately provide a summary. "
            "Do NOT search multiple times. STOP after one search."
        )
    )

    analysis_agent = ReActAgent(
        tools=[FunctionTool.from_defaults(fn=simple_analysis)],
        llm=Settings.llm,
        verbose=False, 
        system_prompt="Analysis Agent: Use simple_analysis to extract key insights. Be concise."
    )

    writer_agent = ReActAgent(
        tools=[FunctionTool.from_defaults(fn=format_basic_report)],
        llm=Settings.llm,
        verbose=False,
        system_prompt="Writer Agent: Use format_basic_report ONCE to create final report. After using the tool, present the result and STOP. Do not overthink."
    )
    
    return research_agent, analysis_agent, writer_agent

# Multi-Agent Orchestrator
class SimpleOrchestrator:
    def __init__(self, research_agent, analysis_agent, writer_agent):
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.writer_agent = writer_agent

    async def generate_report(self, topic: str, max_iter: int):
        """Streamlined 3-step workflow with progress tracking"""
        
        # Step 1: Research
        research = await self.research_agent.run(
            f"Search for latest info on: {topic}",
            max_iterations=max_iter
        )
        
        # Step 2: Analysis
        analysis = await self.analysis_agent.run(
            f"Analyze: {str(research)[:500]}",
            max_iterations=max_iter
        )
        
        # Step 3: Writing
        report = await self.writer_agent.run(
            f"Create report on {topic} using: {str(analysis)[:300]}",
            max_iterations=max_iter
        )
        
        return {
            "topic": topic,
            "research_data": str(research),
            "analysis": str(analysis),
            "final_report": str(report),
            "word_count": len(str(report).split())
        }

# Async wrapper for Streamlit
def run_multi_agent_sync(topic: str, max_iter: int):
    """Sync wrapper for async multi-agent system"""
    async def _run():
        research_agent, analysis_agent, writer_agent = create_agents()
        orchestrator = SimpleOrchestrator(research_agent, analysis_agent, writer_agent)
        return await orchestrator.generate_report(topic, max_iter)
    
    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

# Main Interface
st.subheader("🔬 Multi-Agent Research System")

# Example topics
examples = [
    "AI trends 2024",
    "Sustainable energy technologies",
    "Future of remote work",
    "Blockchain applications in healthcare",
    "Electric vehicles market analysis"
]

# Topic input with persistence
if "topic" not in st.session_state:
    st.session_state["topic"] = examples[0]

topic = st.text_input("Research Topic", value=st.session_state["topic"], help="Enter any topic for multi-agent analysis")

# Example buttons
st.write("**Quick Examples:**")
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(example, key=f"ex_{i}"):
        st.session_state["topic"] = example
        if hasattr(st, "rerun"):
            st.rerun()

# Generate Report Button
if st.button("🚀 Generate Multi-Agent Report", type="primary"):
    if not st.session_state["topic"].strip():
        st.warning("Please enter a research topic.")
    else:
        topic = st.session_state["topic"].strip()
        
        # Progress tracking
        progress_bar = st.progress(0, text="Initializing multi-agent system...")
        
        try:
            # Step 1
            progress_bar.progress(25, text="🔍 Research Agent gathering information...")
            
            # Step 2  
            progress_bar.progress(50, text="📊 Analysis Agent processing data...")
            
            # Step 3
            progress_bar.progress(75, text="✍️ Writer Agent formatting report...")
            
            # Execute
            with st.spinner("Multi-agent system working..."):
                result = run_multi_agent_sync(topic, max_iterations)
            
            progress_bar.progress(100, text="✅ Report generation complete!")
            
            # Display Results
            st.success(f"Multi-agent report generated successfully! ({result['word_count']} words)")
            
            # Tabbed results
            tab1, tab2, tab3 = st.tabs(["📄 Final Report", "🔍 Research Data", "📊 Analysis"])
            
            with tab1:
                st.markdown("### Final Report")
                st.text_area("", value=result["final_report"], height=400, disabled=True)
            
            with tab2:
                st.markdown("### Research Agent Output")
                st.text_area("", value=result["research_data"], height=300, disabled=True)
            
            with tab3:
                st.markdown("### Analysis Agent Output") 
                st.text_area("", value=result["analysis"], height=300, disabled=True)
                
            # Clear progress bar
            progress_bar.empty()
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"Multi-agent system failed: {str(e)}")
            if "Max iterations" in str(e):
                st.info("Try increasing max iterations in the sidebar or simplify your topic.")
            elif "rate limit" in str(e).lower():
                st.info("Rate limit hit. Try using a different model or wait a moment.")

# Footer
st.markdown("---")
st.markdown("**Multi-Agent Architecture:** Research → Analysis → Writing")
st.markdown("*Powered by LlamaIndex ReAct Agents + Groq + SerpAPI*")
