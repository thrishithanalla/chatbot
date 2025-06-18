# chatbot/Notebook/backend/config.py
import os
from dotenv import load_dotenv
import logging
from langchain.prompts import PromptTemplate # Import PromptTemplate

# Load environment variables from .env file in the same directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Environment Variables & Defaults ---

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-r1:1.5b') # Default model for generation/analysis
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'mxbai-embed-large') # Default model for embeddings
OLLAMA_REQUEST_TIMEOUT = int(os.getenv('OLLAMA_REQUEST_TIMEOUT', 180)) # Default 3 minutes

# Nougat OCR Configuration (Optional - for advanced PDF to Markdown conversion)
NOUGAT_ENABLED = os.getenv('NOUGAT_ENABLED', 'False').lower() == 'true'
NOUGAT_API_URL = os.getenv('NOUGAT_API_URL', 'http://localhost:8503/predict') # Default if Nougat server is run locally
NOUGAT_REQUEST_TIMEOUT = int(os.getenv('NOUGAT_REQUEST_TIMEOUT', 300)) # 5 minutes for Nougat processing

# Application Configuration Paths (relative to backend directory)
backend_dir = os.path.dirname(__file__)
FAISS_FOLDER = os.path.join(backend_dir, os.getenv('FAISS_FOLDER', 'faiss_store'))
UPLOAD_FOLDER = os.path.join(backend_dir, os.getenv('UPLOAD_FOLDER', 'uploads'))
DATABASE_NAME = os.getenv('DATABASE_NAME', 'chat_history.db')
DATABASE_PATH = os.path.join(backend_dir, DATABASE_NAME)
DEFAULT_PDFS_FOLDER = os.path.join(backend_dir, os.getenv('DEFAULT_PDFS_FOLDER', 'default_pdfs'))

# File Handling
ALLOWED_EXTENSIONS = {'pdf'}

# RAG Configuration
RAG_CHUNK_K = int(os.getenv('RAG_CHUNK_K', 5)) # Number of unique chunks to finally send to LLM
RAG_SEARCH_K_PER_QUERY = int(os.getenv('RAG_SEARCH_K_PER_QUERY', 3)) # Number of chunks to retrieve per sub-query before deduplication
MULTI_QUERY_COUNT = int(os.getenv('MULTI_QUERY_COUNT', 3)) # Number of sub-questions (0 to disable)

# Analysis Configuration
ANALYSIS_MAX_CONTEXT_LENGTH = int(os.getenv('ANALYSIS_MAX_CONTEXT_LENGTH', 8000)) # Max chars for analysis context

# Web Search Configuration
WEB_SEARCH_ENABLED = os.getenv('WEB_SEARCH_ENABLED', 'True').lower() == 'true'
WEB_SEARCH_MAX_RESULTS = int(os.getenv('WEB_SEARCH_MAX_RESULTS', 3))
WEB_SEARCH_TIMEOUT = int(os.getenv('WEB_SEARCH_TIMEOUT', 10)) # Timeout for DDG search in seconds
WEB_SEARCH_REGION = os.getenv('WEB_SEARCH_REGION', 'wt-wt') # Worldwide
WEB_SEARCH_CACHE_TTL_SECONDS = int(os.getenv('WEB_SEARCH_CACHE_TTL_SECONDS', 3600)) # 1 hour TTL for web search cache

# Chat Configuration
USE_DEEP_THINK_PROMPT = os.getenv('USE_DEEP_THINK_PROMPT', 'True').lower() == 'true' 

# Logging Configuration
LOGGING_LEVEL_NAME = os.getenv('LOGGING_LEVEL', 'INFO').upper()
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_NAME, logging.INFO)
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'


# --- Prompt Templates ---

# Sub-Query Generation Prompt (Thinking optional here, focus on direct output)
SUB_QUERY_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["query", "num_queries"],
    template="""You are an AI assistant skilled at decomposing user questions into effective search queries for a vector database containing chunks of engineering documents.
Given the user's query, generate {num_queries} distinct search queries targeting different specific aspects, keywords, or concepts within the original query.
Focus on creating queries that are likely to retrieve relevant text chunks individually.
Output ONLY the generated search queries, each on a new line. Do not include numbering, labels, explanations, or any other text.

User Query: "{query}"

Generated Search Queries:"""
)


# RAG Synthesis Prompt (Standard - used if DEEP_THINK is False)
SYNTHESIS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["query", "context", "web_search_context"],
    template="""You are a Faculty for engineering students with in-depth knowledge in all engineering subjects and an Expert for an academic audience, ranging from undergraduates to PhD scholars. Your goal is to answer the user's query by synthesizing information from provided document context chunks, web search results (if available), and your general knowledge. Provide detailed, technical, and well-structured responses suitable for this audience. Use precise terminology, include relevant concepts, algorithms, and applications, and organize your response with sections or bullet points where appropriate.

**TASK:** Respond to the user's query using the provided document context, web search results, and your general knowledge. The user will have a separate option to view the raw web search results, so **DO NOT directly quote or list web search snippets in your main answer.**

**USER QUERY:**
"{query}"

**PROVIDED DOCUMENT CONTEXT:**
--- START DOCUMENT CONTEXT ---
{context}
--- END DOCUMENT CONTEXT ---

**WEB SEARCH RESULTS (For your understanding and synthesis. DO NOT quote directly in your answer):**
--- START WEB SEARCH RESULTS ---
{web_search_context}
--- END WEB SEARCH RESULTS ---

**INSTRUCTIONS:**

**STEP 1: THINKING PROCESS (MANDATORY):**
*   **CRITICAL:** Before writing the final answer, articulate your step-by-step reasoning. Explain how you will use the document context, relevant web search results (if any), and general knowledge.
*   Enclose this entire reasoning process *exclusively* within `<thinking>` and `</thinking>` tags.
*   Example: `<thinking>The user asks about X. Document context [1] defines X. Web search results indicate a recent development Y related to X. I will synthesize information from document context [1], incorporate insights about Y from the web search, and supplement with general knowledge about Z, clearly attributing document context sources.</thinking>`
*   **DO NOT** put any text before `<thinking>` or after `</thinking>` except for the final answer.

**STEP 2: FINAL ANSWER (After the `</thinking>` tag):**
*   Provide a comprehensive, synthesized, and helpful answer.
*   **Prioritize Document Context:** Base your answer **primarily** on information within the `PROVIDED DOCUMENT CONTEXT`.
*   **Use Web Search Strategically:**
    *   If document context is insufficient or the query implies a need for very recent/external information, use the `WEB SEARCH RESULTS` to inform your answer.
    *   When web search information is used, you can allude to it generally (e.g., "Recent web information suggests...", "Online sources indicate...", "Broader searches point to...") rather than quoting specific snippets or URLs from the `WEB SEARCH RESULTS` block. The user can view the raw web results separately.
*   **Cite Document Sources:** When using information *directly* from a document context chunk, **you MUST cite** its number like [1], [2], [1][3].
*   **Insufficient Information:** If neither document context, web search, nor general knowledge can answer the query, explicitly state what is missing.
*   **Integrate General Knowledge:** Seamlessly integrate your general knowledge to fill gaps, provide background, or offer broader explanations *after* utilizing context and web search. Clearly signal when you are using general knowledge if it's distinct from the provided contexts.
*   **Be a Tutor:** Explain concepts clearly. Be helpful, accurate, and conversational. Use Markdown formatting for readability.
*   **Accuracy:** Do not invent information. If unsure, state that.

**BEGIN RESPONSE (Start *immediately* with the `<thinking>` tag):**
<thinking>"""
)

# New Deep Think RAG Synthesis Prompt
DEEP_THINK_SYNTHESIS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["query", "context", "web_search_context"],
    template="""You are an advanced AI assistant employing a "Deep Thinking Framework" to provide comprehensive and multi-faceted answers. Your responses are for an academic audience (engineering students to PhD scholars).

**USER QUERY:**
"{query}"

**PROVIDED DOCUMENT CONTEXT (RAG - from internal knowledge base):**
--- START DOCUMENT CONTEXT ---
{context}
--- END DOCUMENT CONTEXT ---
(Use citations like [1], [2] when referencing this context)

**WEB SEARCH RESULTS (External Information):**
--- START WEB SEARCH RESULTS ---
{web_search_context}
--- END WEB SEARCH RESULTS ---
(When referencing, you can state "Web search result [N] suggests..." or "According to [Source URL from web search]...")

**INSTRUCTIONS:**

**STEP 1: DEEP THINKING PROCESS (MANDATORY - Enclose *entirely* within `<thinking>` and `</thinking>` tags):**
Your entire reasoning process MUST be structured into the following three phases. Be thorough and explicit in each phase. This detailed thinking process is for internal use and will be shown to the user *only if they choose to view it*.

**Phase 1: Factual Perspective**
    *   Identify and extract key facts, data, definitions, and specific information relevant to the query from BOTH the "PROVIDED DOCUMENT CONTEXT" and "WEB SEARCH RESULTS".
    *   Clearly attribute sources for each piece of factual information (e.g., "Document context [1] states...", "Web search result [X] from [Source URL if available] indicates...").
    *   Evaluate the reliability or potential biases of web sources if apparent.
    *   If context is missing or conflicting, note this.
    *   Example:
        ```
        **Phase 1: Factual Perspective**
        - Query Core: The query asks about the efficiency of Carnot engines.
        - Document Context Analysis:
            - Document [1] defines a Carnot engine and provides its efficiency formula: η = 1 - (Tc/Th).
            - Document [2] gives an example calculation with Tc=300K, Th=600K, resulting in η=50%.
        - Web Search Analysis:
            - Web Result [1] (source: uni-thermo.edu/carnot) confirms the formula and mentions practical limitations.
            - Web Result [2] (source: randomblog.com/piston-power) makes an unsubstantiated claim of 90% efficiency, which seems unreliable.
        - Factual Summary: Carnot efficiency is theoretically η = 1 - (Tc/Th). Practical engines have lower efficiencies due to real-world irreversibilities.
        ```

**Phase 2: Conceptual Perspective**
    *   Analyze the query within a broader conceptual or theoretical framework.
    *   Identify underlying principles, theories, models, or core concepts that help explain the "why" and "how" behind the facts.
    *   Discuss relationships between concepts, assumptions, and abstract ideas.
    *   Draw connections to established knowledge in the relevant engineering field.
    *   Example:
        ```
        **Phase 2: Conceptual Perspective**
        - Theoretical Framework: The Carnot cycle represents the most efficient possible thermodynamic cycle operating between two temperature reservoirs. This is a consequence of the Second Law of Thermodynamics.
        - Key Concepts: Reversibility, entropy, heat reservoirs, isothermal and adiabatic processes.
        - Relationship to Query: The query about efficiency is fundamentally tied to these thermodynamic limits. Understanding these concepts explains *why* the Carnot efficiency is an upper bound.
        - Assumptions: The ideal Carnot cycle assumes frictionless processes and perfect thermal insulation/conduction as needed, which are not fully achievable.
        ```

**Phase 3: Practical Perspective**
    *   Explore practical applications, real-world implications, examples, limitations, and future considerations related to the query.
    *   Discuss how the concepts and facts translate into engineering practice or observable phenomena.
    *   Consider trade-offs, design challenges, or societal impact if relevant.
    *   Example:
        ```
        **Phase 3: Practical Perspective**
        - Real-world Applications: While no real engine is a perfect Carnot engine, the concept guides the design of power plants (Rankine cycle), refrigeration (reverse Carnot), and internal combustion engines (Otto/Diesel cycles) by setting a benchmark.
        - Limitations: Material properties, heat transfer rates, friction, and working fluid non-idealities prevent real engines from reaching Carnot efficiency.
        - Engineering Implications: Engineers strive to minimize irreversibilities to approach Carnot efficiency for a given temperature range. This involves optimizing materials, lubrication, and heat exchanger design.
        - Future Considerations: Research into new materials or thermodynamic cycles aims to improve efficiencies within these theoretical limits.
        ```

**STEP 2: FINAL SYNTHESIZED ANSWER (After the `</thinking>` tag):**
*   **This is the primary response the user will see.**
*   Based on your three-phase Deep Thinking process (which will be hidden from the user initially), provide a **concise, direct, and well-structured answer** to the user's query: "{query}".
*   Your answer should synthesize the most critical insights from your factual, conceptual, and practical reasoning.
*   **DO NOT repeat the detailed phase headers (e.g., "Factual Perspective:") or the step-by-step breakdown from your thinking process in this final answer.** Instead, integrate the conclusions smoothly.
*   Aim for a clear, helpful, and direct explanation. If the query is "What is Machine Learning?", your answer should directly define and explain Machine Learning, drawing upon your deeper thinking.
*   Use precise terminology. Provide detailed, technical explanations suitable for the academic audience, but ensure the answer is focused and directly addresses the query.
*   Cite document context sources like [1], [2] where essential for key facts in the final answer. You can generally refer to web search insights (e.g., "current research suggests...") rather than specific web result numbers here.
*   If confidence is low due to limited or conflicting information from your thinking process, state this (e.g., "Based on available information, which is limited, it appears that...").
*   If information cannot be verified, include a disclaimer (e.g., "This aspect could not be verified from the provided document context or web search results.").
*   Use Markdown for clear formatting (paragraphs, lists, bolding as appropriate for a direct answer).

**BEGIN RESPONSE (Start *immediately* with the `<thinking>` tag):**
<thinking>
*(Your detailed Factual, Conceptual, and Practical phase analysis here, as outlined above)*
</thinking>
*(Your CONCISE, DIRECT, and FINAL synthesized answer to "{query}" here, integrating insights from the thinking process. DO NOT include phase headers. This is the primary answer the user sees.)*
"""
)


# Analysis Prompts (Thinking Recommended)
_ANALYSIS_THINKING_PREFIX = """**STEP 1: THINKING PROCESS (Recommended):**
*   Before generating the analysis, briefly outline your plan in `<thinking>` tags. Example: `<thinking>Analyzing for FAQs. Will scan for key questions and answers presented in the text.</thinking>`
*   If you include thinking, place the final analysis *after* the `</thinking>` tag.

**STEP 2: ANALYSIS OUTPUT:**
*   Generate the requested analysis based **strictly** on the text provided below.
*   Follow the specific OUTPUT FORMAT instructions carefully.

--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
"""

ANALYSIS_PROMPTS = {
    "faq": PromptTemplate(
        input_variables=["doc_text_for_llm"],
        template=_ANALYSIS_THINKING_PREFIX + """
**TASK:** Generate 5-7 Frequently Asked Questions (FAQs) with concise answers based ONLY on the text.

**OUTPUT FORMAT (Strict):**
*   Start directly with the first FAQ (after thinking, if used). Do **NOT** include preamble.
*   Format each FAQ as:
    Q: [Question derived ONLY from the text]
    A: [Answer derived ONLY from the text, concise]
*   If the text doesn't support an answer, don't invent one. Use Markdown for formatting if appropriate (e.g., lists within an answer).

**BEGIN OUTPUT (Start with 'Q:' or `<thinking>`):**
"""
    ),
    "topics": PromptTemplate(
        input_variables=["doc_text_for_llm"],
        template=_ANALYSIS_THINKING_PREFIX + """
**TASK:** Identify the 5-8 most important topics discussed. Provide a 1-2 sentence explanation per topic based ONLY on the text.

**OUTPUT FORMAT (Strict):**
*   Start directly with the first topic (after thinking, if used). Do **NOT** include preamble.
*   Format as a Markdown bulleted list:
    *   **Topic Name:** Brief explanation derived ONLY from the text content (1-2 sentences max).

**BEGIN OUTPUT (Start with '*   **' or `<thinking>`):**
"""
    ),
    "mindmap": PromptTemplate(
        input_variables=["doc_text_for_llm"],
        template=_ANALYSIS_THINKING_PREFIX + """
**TASK:** You are an AI assistant tasked with creating a hierarchical mind map outline from the provided document text.
The outline should use **Markdown headings (e.g., #, ##, ###) and lists (e.g., - )** to represent the structure of concepts.
**IMPORTANT:** The hierarchy of your Markdown output should directly reflect the structure, sections, headings, and key listed items found in the 'DOCUMENT TEXT'. Do not invent new top-level categories unless they are explicitly named as such in the document. If the document is a flat list of features or topics under a main title, represent that structure.

(Note: The input 'DOCUMENT TEXT' might be structured Markdown if extracted via Nougat. Leverage its existing headings and lists if present for a better mind map outline. If it's plain text, identify the main concepts and their relationships as presented.)

**OUTPUT FORMAT (Strict):**
*   **CRITICAL INSTRUCTION:** If you include a thinking process within `<thinking>...</thinking>` tags, the Markdown mind map structure MUST start on a new line *immediately after* the closing `</thinking>` tag. Do NOT include any other text or preamble between the `</thinking>` tag and the first Markdown heading. If no thinking process is included, start directly with the first Markdown heading (e.g., `# Document Title` or `# Main Feature`).
*   The Markdown outline should be the *only* content after the optional thinking block.
*   Use Markdown headings (#, ##, ###, etc.) for different levels of the hierarchy.
*   Use bullet points (-) or numbered lists for items or sub-details under headings.
*   Ensure a logical and hierarchical structure that closely follows the content of the 'DOCUMENT TEXT'.

*   **Example 1 (If document has clear sections):**
    ```markdown
    # Document Main Title
    ## Section 1: Introduction
    - Key point 1.1
    - Key point 1.2
    ## Section 2: Core Concepts
    ### Sub-Concept A
    - Detail A.1
    ### Sub-Concept B
    - Detail B.1
    ```

*   **Example 2 (If document is a list of features under a title):**
    Suppose Document Text is:
    ```
    AI Capabilities
    1. Learning
    2. Reasoning
       - Deductive
       - Inductive
    3. Perception
    ```
    Your Markdown output should be:
    ```markdown
    # AI Capabilities
    ## Learning
    ## Reasoning
    - Deductive
    - Inductive
    ## Perception
    ```
    Alternatively, using bullets for the main features:
    ```markdown
    # AI Capabilities
    - Learning
    - Reasoning
      - Deductive
      - Inductive
    - Perception
    ```

**BEGIN OUTPUT (Follow critical instructions above regarding thinking block and Markdown placement):**
"""
    )
}


# --- Logging Setup ---
def setup_logging():
    """Configures application-wide logging."""
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    # Suppress excessive logging from noisy libraries if necessary
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING) # FAISS can be verbose
    logging.getLogger("duckduckgo_search").setLevel(logging.INFO) # Or WARNING if too verbose
    # Add more loggers to suppress as needed

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {LOGGING_LEVEL_NAME}")
    logger.debug(f"OLLAMA_BASE_URL={OLLAMA_BASE_URL}")
    logger.debug(f"OLLAMA_MODEL={OLLAMA_MODEL}")
    logger.debug(f"OLLAMA_EMBED_MODEL={OLLAMA_EMBED_MODEL}")
    logger.debug(f"NOUGAT_ENABLED={NOUGAT_ENABLED}, NOUGAT_API_URL={NOUGAT_API_URL}, NOUGAT_REQUEST_TIMEOUT={NOUGAT_REQUEST_TIMEOUT}s")
    logger.debug(f"FAISS_FOLDER={FAISS_FOLDER}")
    logger.debug(f"UPLOAD_FOLDER={UPLOAD_FOLDER}")
    logger.debug(f"DATABASE_PATH={DATABASE_PATH}")
    logger.debug(f"RAG_CHUNK_K={RAG_CHUNK_K}, RAG_SEARCH_K_PER_QUERY={RAG_SEARCH_K_PER_QUERY}, MULTI_QUERY_COUNT={MULTI_QUERY_COUNT}")
    logger.debug(f"ANALYSIS_MAX_CONTEXT_LENGTH={ANALYSIS_MAX_CONTEXT_LENGTH}")
    logger.debug(f"WEB_SEARCH_ENABLED={WEB_SEARCH_ENABLED}, WEB_SEARCH_MAX_RESULTS={WEB_SEARCH_MAX_RESULTS}, WEB_SEARCH_TIMEOUT={WEB_SEARCH_TIMEOUT}s, WEB_SEARCH_REGION={WEB_SEARCH_REGION}, WEB_SEARCH_CACHE_TTL_SECONDS={WEB_SEARCH_CACHE_TTL_SECONDS}s")
    logger.debug(f"USE_DEEP_THINK_PROMPT={USE_DEEP_THINK_PROMPT}")