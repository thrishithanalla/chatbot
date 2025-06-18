# chatbot/Notebook/backend/ai_core.py
# --- START OF FILE ai_core.py ---

# Notebook/backend/ai_core.py
import os
import logging
import fitz  # PyMuPDF
import re
import requests # For calling Nougat API
import time # For web search cache

# Near the top of ai_core.py
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
# Removed incorrect OllamaLLM import if it was there from previous attempts
from langchain.text_splitter import RecursiveCharacterTextSplitter # <<<--- ENSURE THIS IS PRESENT
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate # Import PromptTemplate if needed directly here
from duckduckgo_search import DDGS # For web search

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL, FAISS_FOLDER,
    DEFAULT_PDFS_FOLDER, UPLOAD_FOLDER, RAG_CHUNK_K, MULTI_QUERY_COUNT,
    ANALYSIS_MAX_CONTEXT_LENGTH, OLLAMA_REQUEST_TIMEOUT, RAG_SEARCH_K_PER_QUERY,
    SUB_QUERY_PROMPT_TEMPLATE, SYNTHESIS_PROMPT_TEMPLATE, ANALYSIS_PROMPTS,
    WEB_SEARCH_ENABLED, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT, WEB_SEARCH_REGION, # Web search config
    NOUGAT_ENABLED, NOUGAT_API_URL, NOUGAT_REQUEST_TIMEOUT, # Nougat config
    USE_DEEP_THINK_PROMPT, DEEP_THINK_SYNTHESIS_PROMPT_TEMPLATE, # Deep Think config
    WEB_SEARCH_CACHE_TTL_SECONDS # Web search cache config
)
from utils import parse_llm_response, escape_html # Added escape_html for potential use

logger = logging.getLogger(__name__)

# --- Global State (managed within functions) ---
document_texts_cache = {}
vector_store = None
embeddings: OllamaEmbeddings | None = None
llm: ChatOllama | None = None
web_search_cache = {} # Simple in-memory cache for web search: {query: (timestamp, results_text)}


# --- Initialization Functions ---

# ai_core.py (only showing the modified function)
def initialize_ai_components() -> tuple[OllamaEmbeddings | None, ChatOllama | None]:
    """Initializes Ollama Embeddings and LLM instances globally.

    Returns:
        tuple[OllamaEmbeddings | None, ChatOllama | None]: The initialized embeddings and llm objects,
                                                          or (None, None) if initialization fails.
    """
    global embeddings, llm
    if embeddings and llm:
        logger.info("AI components already initialized.")
        return embeddings, llm

    try:
        # Use the new OllamaEmbeddings from langchain_ollama
        logger.info(f"Initializing Ollama Embeddings: model={OLLAMA_EMBED_MODEL}, base_url={OLLAMA_BASE_URL}, timeout={OLLAMA_REQUEST_TIMEOUT}s")
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            #request_timeout=OLLAMA_REQUEST_TIMEOUT # Explicitly pass timeout
        )
        # Perform a quick test embedding
        _ = embeddings.embed_query("Test embedding query")
        logger.info("Ollama Embeddings initialized successfully.")

        # Use the new ChatOllama from langchain_ollama
        logger.info(f"Initializing Ollama LLM: model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}, timeout={OLLAMA_REQUEST_TIMEOUT}s")
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            #request_timeout=OLLAMA_REQUEST_TIMEOUT # Explicitly pass timeout
        )
        # Perform a quick test invocation
        _ = llm.invoke("Respond briefly with 'AI Check OK'")
        logger.info("Ollama LLM initialized successfully.")

        return embeddings, llm  # Return the objects
    except ImportError as e:
        logger.critical(f"Import error during AI initialization: {e}. Ensure correct langchain packages are installed.", exc_info=True)
        embeddings = None
        llm = None
        return None, None
    except Exception as e:
        # Catch potential Pydantic validation error specifically if possible, or general Exception
        logger.error(f"Failed to initialize AI components (check Ollama server status, model name '{OLLAMA_MODEL}' / '{OLLAMA_EMBED_MODEL}', base URL '{OLLAMA_BASE_URL}', timeout {OLLAMA_REQUEST_TIMEOUT}s): {e}", exc_info=True)
        # Log the type of error for better debugging
        logger.error(f"Error Type: {type(e).__name__}")
        # If it's a Pydantic error, the message usually contains details
        if "pydantic" in str(type(e)).lower():
             logger.error(f"Pydantic Validation Error Details: {e}")
        embeddings = None
        llm = None
        return None, None

def load_vector_store() -> bool:
    """Loads the FAISS index from disk into the global `vector_store`.

    Requires `embeddings` to be initialized first.

    Returns:
        bool: True if the index was loaded successfully, False otherwise (or if not found).
    """
    global vector_store, embeddings
    if vector_store:
        logger.info("Vector store already loaded.")
        return True
    if not embeddings:
        logger.error("Embeddings not initialized. Cannot load vector store.")
        return False

    faiss_index_path = os.path.join(FAISS_FOLDER, "index.faiss")
    faiss_pkl_path = os.path.join(FAISS_FOLDER, "index.pkl")

    if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
        try:
            logger.info(f"Loading FAISS index from folder: {FAISS_FOLDER}")
            # Note: Loading requires the same embedding model used for saving.
            # allow_dangerous_deserialization is required for FAISS/pickle
            vector_store = FAISS.load_local(
                folder_path=FAISS_FOLDER,
                embeddings=embeddings, # Pass the initialized embeddings object
                allow_dangerous_deserialization=True
            )
            index_size = getattr(getattr(vector_store, 'index', None), 'ntotal', 0)
            if index_size > 0:
                logger.info(f"FAISS index loaded successfully. Contains {index_size} vectors.")
                return True
            else:
                logger.warning(f"FAISS index loaded from {FAISS_FOLDER}, but it appears to be empty.")
                return True # Treat empty as loaded
        except FileNotFoundError:
            logger.warning(f"FAISS index files not found in {FAISS_FOLDER}, although directory exists. Proceeding without loaded index.")
            vector_store = None
            return False
        except EOFError:
            logger.error(f"EOFError loading FAISS index from {FAISS_FOLDER}. Index file might be corrupted or incomplete.", exc_info=True)
            vector_store = None
            return False
        except Exception as e:
            logger.error(f"Error loading FAISS index from {FAISS_FOLDER}: {e}", exc_info=True)
            vector_store = None # Ensure it's None if loading failed
            return False
    else:
        logger.warning(f"FAISS index files (index.faiss, index.pkl) not found at {FAISS_FOLDER}. Will be created on first upload or if default.py ran.")
        vector_store = None
        return False # Indicate index wasn't loaded


def save_vector_store() -> bool:
    """Saves the current global `vector_store` (FAISS index) to disk.

    Returns:
        bool: True if saving was successful, False otherwise (or if store is None).
    """
    global vector_store
    if not vector_store:
        logger.warning("Attempted to save vector store, but it's not loaded or initialized.")
        return False
    if not os.path.exists(FAISS_FOLDER):
        try:
            os.makedirs(FAISS_FOLDER)
            logger.info(f"Created FAISS store directory: {FAISS_FOLDER}")
        except OSError as e:
            logger.error(f"Failed to create FAISS store directory {FAISS_FOLDER}: {e}", exc_info=True)
            return False

    try:
        index_size = getattr(getattr(vector_store, 'index', None), 'ntotal', 0)
        logger.info(f"Saving FAISS index ({index_size} vectors) to {FAISS_FOLDER}...")
        vector_store.save_local(FAISS_FOLDER)
        logger.info(f"FAISS index saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving FAISS index to {FAISS_FOLDER}: {e}", exc_info=True)
        return False


def load_all_document_texts():
    """Loads text from all PDFs found in default and upload folders into the global cache.
    Uses PyMuPDF by default for caching. Nougat is used on-demand for specific analyses.
    """
    global document_texts_cache
    logger.info("Loading/refreshing document texts cache for analysis (using PyMuPDF)...")
    document_texts_cache = {} # Reset cache before loading
    loaded_count = 0
    processed_files = set()

    def _load_from_folder(folder_path):
        nonlocal loaded_count
        count = 0
        if not os.path.exists(folder_path):
            logger.warning(f"Document text folder not found: {folder_path}. Skipping.")
            return count
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith('.pdf') and not filename.startswith('~') and filename not in processed_files:
                    file_path = os.path.join(folder_path, filename)
                    # For cache, always use PyMuPDF for speed and general use.
                    text = extract_text_from_pdf(file_path, prefer_nougat=False)
                    if text:
                        document_texts_cache[filename] = text
                        processed_files.add(filename)
                        count += 1
                    else:
                        logger.warning(f"Could not extract text from {filename} in {folder_path} for cache (PyMuPDF).")
            logger.info(f"Cached text for {count} PDFs from {folder_path} (PyMuPDF).")
            loaded_count += count
        except Exception as e:
            logger.error(f"Error listing or processing files in {folder_path} for cache: {e}", exc_info=True)
        return count

    _load_from_folder(DEFAULT_PDFS_FOLDER)
    _load_from_folder(UPLOAD_FOLDER)
    logger.info(f"Finished loading texts cache. Total unique documents cached: {len(document_texts_cache)}")


# --- PDF Processing Functions ---

def _invoke_nougat_service(pdf_path: str) -> str | None:
    """
    Helper function to call a Nougat service to extract structured Markdown.
    Requires a running Nougat API service (e.g., from facebookresearch/nougat/app.py or nougat_api.py).
    """
    logger.info(f"Attempting to process PDF '{os.path.basename(pdf_path)}' with Nougat service at {NOUGAT_API_URL} (timeout: {NOUGAT_REQUEST_TIMEOUT}s)...")
    if not os.path.exists(pdf_path): # Should be checked by caller, but good for direct use.
        logger.error(f"Nougat processing: PDF file not found at {pdf_path}")
        return None

    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            response = requests.post(NOUGAT_API_URL, files=files, timeout=NOUGAT_REQUEST_TIMEOUT)

        response.raise_for_status() # Raise an exception for HTTP errors (4xx, 5xx)
        
        markdown_content = response.text # Nougat API typically returns raw MMD content
        if markdown_content:
            logger.info(f"Successfully extracted Markdown from '{os.path.basename(pdf_path)}' using Nougat (approx {len(markdown_content)} chars).")
            return markdown_content.strip() # Return MMD content
        else:
            logger.warning(f"Nougat service returned empty content for '{os.path.basename(pdf_path)}'. This might be valid for empty/image-only PDFs.")
            return "" # Return empty string if Nougat provides no content but succeeds
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Nougat service at {NOUGAT_API_URL}. Ensure Nougat API server is running and accessible.")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Request to Nougat service timed out for '{os.path.basename(pdf_path)}' after {NOUGAT_REQUEST_TIMEOUT}s.")
        return None
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text[:200] if hasattr(e.response, 'text') else "No additional error text."
        logger.error(f"Nougat service returned an HTTP error for '{os.path.basename(pdf_path)}': {e.response.status_code} - {error_text}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while invoking Nougat service for '{os.path.basename(pdf_path)}': {e}", exc_info=True)
        return None

def extract_text_from_pdf(pdf_path: str, prefer_nougat: bool = False) -> str | None:
    """
    Extracts text from a single PDF file.
    If prefer_nougat is True and Nougat is enabled and configured, attempts Nougat.
    If Nougat fails, is not preferred, or not enabled/configured, falls back to PyMuPDF.

    Args:
        pdf_path (str): The full path to the PDF file.
        prefer_nougat (bool): If True, attempt Nougat extraction first.

    Returns:
        str | None: Extracted text (Markdown if Nougat, plain if PyMuPDF).
                    Returns "" if extraction is successful but yields no text.
                    Returns None if a critical error occurs during extraction.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found for extraction: {pdf_path}")
        return None

    text_content: str | None = None
    extraction_method_tried = "None"

    if prefer_nougat and NOUGAT_ENABLED and NOUGAT_API_URL:
        extraction_method_tried = "Nougat"
        logger.debug(f"Nougat extraction preferred and configured for '{os.path.basename(pdf_path)}'.")
        text_content = _invoke_nougat_service(pdf_path)
        if text_content is not None: # Nougat call completed (successfully or returned empty string)
            if text_content == "":
                logger.warning(f"Nougat returned empty text for '{os.path.basename(pdf_path)}'. This is treated as successful empty extraction.")
            return text_content # Return Nougat's output (Markdown or empty string)
        else: # Nougat call failed (returned None)
            logger.warning(f"Nougat extraction failed for '{os.path.basename(pdf_path)}'. Falling back to PyMuPDF.")
            # Fall through to PyMuPDF
    elif prefer_nougat: # Nougat preferred but not enabled/configured
        logger.debug(f"Nougat extraction preferred but Nougat not enabled/configured. Using PyMuPDF for '{os.path.basename(pdf_path)}'.")

    # PyMuPDF (fitz) extraction - fallback or default
    extraction_method_tried = "PyMuPDF" if extraction_method_tried == "None" else "PyMuPDF (fallback)"
    logger.debug(f"Attempting PyMuPDF extraction for '{os.path.basename(pdf_path)}' ({extraction_method_tried}).")
    text = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        # logger.debug(f"Starting PyMuPDF text extraction from {os.path.basename(pdf_path)} ({num_pages} pages)...")
        for page_num in range(num_pages):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text("text", sort=True, flags=0).strip()
                page_text = re.sub(r'[ \t\f\v]+', ' ', page_text) 
                page_text = re.sub(r'\n+', '\n', page_text) 
                if page_text:
                    text += page_text + "\n\n" 
            except Exception as page_err:
                logger.warning(f"Error processing page {page_num+1} of {os.path.basename(pdf_path)} with PyMuPDF: {page_err}")
                continue
        doc.close()
        cleaned_text = text.strip()
        if cleaned_text:
            logger.info(f"Successfully extracted text from {os.path.basename(pdf_path)} using PyMuPDF (approx {len(cleaned_text)} chars).")
            return cleaned_text
        else:
            logger.warning(f"PyMuPDF extracted text was empty for {os.path.basename(pdf_path)}. This is treated as successful empty extraction.")
            return "" # Return empty string if PyMuPDF extracted nothing
    except fitz.fitz.PasswordError:
        logger.error(f"Error extracting text with PyMuPDF from {os.path.basename(pdf_path)}: File is password-protected.")
        return None # Password error is a hard failure for PyMuPDF
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF from {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return None # Other PyMuPDF errors are hard failures


def create_chunks_from_text(text: str, filename: str) -> list[Document]:
    """Splits text into chunks using RecursiveCharacterTextSplitter and creates LangChain Documents.
    Handles both plain text and Markdown input.
    """
    if not text: # Handles None or empty string
        logger.warning(f"Cannot create chunks for '{filename}', input text is empty.")
        return []

    # RecursiveCharacterTextSplitter is generally good with Markdown too due to its separator list.
    # If Nougat provides highly structured Markdown, more specialized Markdown splitters could be considered,
    # but RecursiveCharacterTextSplitter is a robust default.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Target size of each chunk
        chunk_overlap=150,    # Overlap between chunks
        length_function=len,
        add_start_index=True, # Include start index in metadata
        separators=["\n\n", "\n", ". ", ", ", " ", ""], # Hierarchical separators
        is_separator_regex=False, # Treat separators literally
    )

    try:
        # Use create_documents which handles metadata assignment more cleanly
        documents = text_splitter.create_documents([text], metadatas=[{"source": filename}])
        # Add explicit chunk_index for clarity (though start_index is also present)
        for i, doc in enumerate(documents):
            doc.metadata["chunk_index"] = i

        logger.info(f"Created {len(documents)} LangChain Document chunks for '{filename}'.")
        return documents

    except Exception as e:
        logger.error(f"Error creating chunks for '{filename}': {e}", exc_info=True)
        return []

def add_documents_to_vector_store(documents: list[Document]) -> bool:
    """Adds LangChain Documents to the global FAISS index.
    Creates the index if it doesn't exist. Saves the index afterwards.

    Args:
        documents (list[Document]): The list of documents to add.

    Returns:
        bool: True if documents were added and the index saved successfully, False otherwise.
    """
    global vector_store, embeddings
    if not documents:
        logger.warning("No documents provided to add to vector store.")
        return True # Nothing to add, technically successful no-op.
    if not embeddings:
        logger.error("Embeddings not initialized. Cannot add documents to vector store.")
        return False

    try:
        if vector_store:
            logger.info(f"Adding {len(documents)} document chunks to existing FAISS index...")
            vector_store.add_documents(documents)
            index_size = getattr(getattr(vector_store, 'index', None), 'ntotal', 0)
            logger.info(f"Addition complete. Index now contains {index_size} vectors.")
        else:
            logger.info(f"No FAISS index loaded. Creating new index from {len(documents)} document chunks...")
            vector_store = FAISS.from_documents(documents, embeddings)
            index_size = getattr(getattr(vector_store, 'index', None), 'ntotal', 0)
            if vector_store and index_size > 0:
                logger.info(f"New FAISS index created with {index_size} vectors.")
            else:
                logger.error("Failed to create new FAISS index or index is empty after creation.")
                vector_store = None # Ensure it's None if creation failed
                return False

        # IMPORTANT: Persist the updated index
        return save_vector_store()

    except Exception as e:
        logger.error(f"Error adding documents to FAISS index or saving: {e}", exc_info=True)
        # Consider state: if vector_store existed before, it might be partially updated in memory.
        # Saving failed, so on next load, it should revert unless error was in 'from_documents'.
        return False

# --- RAG and LLM Interaction ---

def generate_sub_queries(query: str) -> list[str]:
    """
    Uses the LLM to generate sub-queries for RAG. Includes the original query.
    Uses SUB_QUERY_PROMPT_TEMPLATE from config.
    """
    global llm
    if not llm:
        logger.error("LLM not initialized, cannot generate sub-queries. Using original query only.")
        return [query]
    if MULTI_QUERY_COUNT <= 0:
        logger.debug("MULTI_QUERY_COUNT is <= 0, skipping sub-query generation.")
        return [query]

    # Use the prompt template from config
    chain = LLMChain(llm=llm, prompt=SUB_QUERY_PROMPT_TEMPLATE)

    try:
        logger.info(f"Generating {MULTI_QUERY_COUNT} sub-queries for: '{query[:100]}...'")
        prompt_to_log = SUB_QUERY_PROMPT_TEMPLATE.format(query=query, num_queries=MULTI_QUERY_COUNT)
        logger.debug(f"Sub-query Prompt (Start):\n{prompt_to_log[:150]}...")

        response = chain.invoke({"query": query, "num_queries": MULTI_QUERY_COUNT})
        raw_response_text = response.get('text', '') if isinstance(response, dict) else str(response)
        logger.debug(f"Sub-query Raw Response (Start):\n{raw_response_text[:150]}...")

        sub_queries = [q.strip() for q in raw_response_text.strip().split('\n') if q.strip()]

        if sub_queries:
            logger.info(f"Generated {len(sub_queries)} sub-queries.")
            final_queries = [query] + sub_queries[:MULTI_QUERY_COUNT]
            final_queries = list(dict.fromkeys(final_queries))
            logger.debug(f"Final search queries: {final_queries}")
            return final_queries
        else:
            logger.warning("LLM did not generate any valid sub-queries. Falling back to original query only.")
            return [query]

    except Exception as e:
        logger.error(f"Error generating sub-queries: {e}", exc_info=True)
        return [query] # Fallback

def perform_rag_search(query: str) -> tuple[list[Document], str, dict[int, dict]]:
    """
    Performs RAG: generates sub-queries, searches vector store, deduplicates, formats context, creates citation map.
    """
    global vector_store
    context_docs = []
    formatted_context_text = "No relevant context was found in the available documents."
    context_docs_map = {} # Use 1-based index for keys mapping to doc details

    if not vector_store:
        logger.warning("RAG search attempted but no vector store is loaded.")
        return context_docs, formatted_context_text, context_docs_map
    if not query or not query.strip():
        logger.warning("RAG search attempted with empty query.")
        return context_docs, formatted_context_text, context_docs_map

    index_size = getattr(getattr(vector_store, 'index', None), 'ntotal', 0)
    if index_size == 0:
        logger.warning("RAG search attempted but the vector store index is empty.")
        return context_docs, formatted_context_text, context_docs_map

    try:
        search_queries = generate_sub_queries(query)
        all_retrieved_docs_with_scores = []
        k_per_query = max(RAG_SEARCH_K_PER_QUERY, 1)
        logger.debug(f"Retrieving top {k_per_query} chunks for each of {len(search_queries)} queries.")

        for q_idx, q in enumerate(search_queries):
            try:
                retrieved = vector_store.similarity_search_with_score(q, k=k_per_query)
                all_retrieved_docs_with_scores.extend(retrieved)
                logger.debug(f"Query {q_idx+1}/{len(search_queries)} ('{q[:50]}...') retrieved {len(retrieved)} chunks.")
            except Exception as search_err:
                logger.error(f"Error during similarity search for query '{q[:50]}...': {search_err}", exc_info=False)

        if not all_retrieved_docs_with_scores:
            logger.info("No relevant chunks found in vector store for the query/sub-queries.")
            return context_docs, formatted_context_text, context_docs_map

        unique_docs_dict = {}
        for doc, score in all_retrieved_docs_with_scores:
            source = doc.metadata.get('source', 'Unknown')
            chunk_idx = doc.metadata.get('chunk_index', doc.metadata.get('start_index', -1))
            doc_key = (source, chunk_idx)
            if doc_key not in unique_docs_dict or score < unique_docs_dict[doc_key][1]:
                unique_docs_dict[doc_key] = (doc, score)

        sorted_unique_docs = sorted(unique_docs_dict.values(), key=lambda item: item[1])
        final_context_docs_with_scores = sorted_unique_docs[:RAG_CHUNK_K]
        context_docs = [doc for doc, score in final_context_docs_with_scores]
        logger.info(f"Retrieved {len(all_retrieved_docs_with_scores)} chunks total. Selected {len(context_docs)} unique chunks.")

        formatted_context_parts = []
        temp_map = {}
        for i, doc in enumerate(context_docs):
            citation_index = i + 1
            source = doc.metadata.get('source', 'Unknown Source')
            chunk_idx = doc.metadata.get('chunk_index', 'N/A')
            content = doc.page_content
            context_str = f"[{citation_index}] Source: {source} | Chunk Index: {chunk_idx}\n{content}"
            formatted_context_parts.append(context_str)
            temp_map[citation_index] = {
                "source": source,
                "chunk_index": chunk_idx,
                "content": content
            }
        formatted_context_text = "\n\n---\n\n".join(formatted_context_parts) if formatted_context_parts else "No context chunks selected."
        context_docs_map = temp_map

    except Exception as e:
        logger.error(f"Error during RAG search for query '{query[:50]}...': {e}", exc_info=True)
        context_docs, formatted_context_text, context_docs_map = [], "Error retrieving context.", {}

    return context_docs, formatted_context_text, context_docs_map

def perform_web_search(query: str) -> str | None:
    """
    Performs a web search using DuckDuckGo and formats results for the LLM.
    Includes a simple time-based cache.
    """
    global web_search_cache
    if not WEB_SEARCH_ENABLED:
        logger.debug("Web search is disabled in config. Skipping.")
        return None

    # Check cache first
    cached_entry = web_search_cache.get(query)
    if cached_entry:
        timestamp, results_text = cached_entry
        if (time.time() - timestamp) < WEB_SEARCH_CACHE_TTL_SECONDS:
            logger.info(f"Returning cached web search results for: '{query[:100]}...'")
            return results_text
        else:
            logger.info(f"Cached web search results for '{query[:100]}...' expired. Performing new search.")
            del web_search_cache[query] # Remove expired entry
    
    logger.info(f"Performing web search for: '{query[:100]}...'")
    try:
        with DDGS(timeout=WEB_SEARCH_TIMEOUT) as ddgs:
            results = ddgs.text(
                keywords=query,
                region=WEB_SEARCH_REGION,
                safesearch='moderate',
                max_results=WEB_SEARCH_MAX_RESULTS
            )
            if not results:
                logger.info("Web search returned no results.")
                web_search_cache[query] = (time.time(), None) # Cache negative result
                return None

            formatted_results = []
            for i, res in enumerate(results):
                title = res.get('title', 'No Title')
                href = res.get('href', '#')
                body = res.get('body', 'No snippet available.')
                # Truncate body for conciseness in prompt
                body_preview = body[:250] + "..." if len(body) > 250 else body
                formatted_results.append(f"Web Search Result [{i+1}]: {title}\nSource: {href}\nSnippet: {body_preview.strip()}")
            
            final_text = "\n\n---\n\n".join(formatted_results)
            logger.info(f"Web search yielded {len(results)} results. Formatted text length: {len(final_text)}")
            web_search_cache[query] = (time.time(), final_text) # Cache successful result
            return final_text

    except Exception as e:
        logger.error(f"Error during web search for query '{query[:100]}...': {e}", exc_info=True)
        return None


def synthesize_chat_response(query: str, context_text: str, web_search_results_text: str | None) -> tuple[str, str | None]:
    """
    Generates the final chat response using the LLM, query, RAG context, and web search results.
    Requests and parses thinking/reasoning content.
    Uses DEEP_THINK_SYNTHESIS_PROMPT_TEMPLATE if USE_DEEP_THINK_PROMPT is True, otherwise SYNTHESIS_PROMPT_TEMPLATE.

    Returns:
        tuple[str, str | None]: (user_answer, thinking_content)
    """
    global llm
    if not llm:
        logger.error("LLM not initialized, cannot synthesize response.")
        return "Error: The AI model is currently unavailable.", None

    WEB_SEARCH_NOT_PERFORMED_MESSAGE = "Web search was not performed or yielded no relevant results for this query."
    web_search_context_for_prompt = web_search_results_text if web_search_results_text else WEB_SEARCH_NOT_PERFORMED_MESSAGE

    chosen_prompt_template = DEEP_THINK_SYNTHESIS_PROMPT_TEMPLATE if USE_DEEP_THINK_PROMPT else SYNTHESIS_PROMPT_TEMPLATE
    prompt_name = "DeepThinkSynthesis" if USE_DEEP_THINK_PROMPT else "StandardSynthesis"

    try:
        final_prompt = chosen_prompt_template.format(
            query=query,
            context=context_text,
            web_search_context=web_search_context_for_prompt
        )
        logger.info(f"Sending {prompt_name} prompt to LLM (model: {OLLAMA_MODEL})...")
        # Log more chars if needed, especially web context part and the multi-phase structure for deep think
        log_prompt_length = 500 if USE_DEEP_THINK_PROMPT else 300
        logger.debug(f"{prompt_name} Prompt (Start):\n{final_prompt[:log_prompt_length]}...")

    except KeyError as e:
        logger.error(f"Error formatting {prompt_name}PromptTemplate: Missing key {e}. Check config.py and prompt variables.")
        return "Error: Internal prompt configuration issue.", None
    except Exception as e:
         logger.error(f"Error creating {prompt_name} prompt: {e}", exc_info=True)
         return "Error: Could not prepare the request for the AI model.", None

    try:
        response_object = llm.invoke(final_prompt)
        full_llm_response = getattr(response_object, 'content', str(response_object))

        logger.info(f"LLM {prompt_name} response received (length: {len(full_llm_response)}).")
        logger.debug(f"{prompt_name} Raw Response (Start):\n{full_llm_response[:300]}...") # Increased for deep think

        user_answer, thinking_content = parse_llm_response(full_llm_response)

        if thinking_content:
            logger.info(f"Parsed thinking content (length: {len(thinking_content)}).")
            if USE_DEEP_THINK_PROMPT:
                 logger.debug(f"DeepThink Reasoning (first 300 chars):\n{thinking_content[:300]}...")
        else:
            logger.debug("No <thinking> content found or parsed in the LLM response.")

        if not user_answer and thinking_content:
             logger.warning("Parsed user answer is empty after removing thinking block.")
             user_answer = "[AI response consisted only of reasoning. No final answer provided. See thinking process.]"
        elif not user_answer and not thinking_content:
             logger.error("LLM response parsing resulted in empty answer and no thinking content.")
             user_answer = "[AI Response Processing Error: Empty result after parsing]"

        if user_answer.strip().startswith("Error:") or "sorry, I encountered an error" in user_answer.lower():
            logger.warning(f"LLM {prompt_name} seems to have resulted in an error message: '{user_answer[:100]}...'")

        return user_answer.strip(), thinking_content

    except Exception as e:
        logger.error(f"LLM chat {prompt_name} failed: {e}", exc_info=True)
        error_message = f"Sorry, I encountered an error while generating the response ({type(e).__name__})."
        return error_message, None


def generate_document_analysis(filename: str, analysis_type: str) -> tuple[str | None, str | None]:
    """
    Generates analysis (FAQ, Topics, Mindmap) for a specific document.
    For 'mindmap' analysis, attempts to use Nougat for PDF-to-Markdown conversion if enabled.
    Otherwise, uses cached PyMuPDF text or live PyMuPDF extraction.

    Returns:
        tuple[str | None, str | None]: (analysis_content, thinking_content)
                                    Returns (error_message, thinking_content) on failure.
                                    Returns (None, None) if document text cannot be found/loaded.
    """
    global llm, document_texts_cache
    logger.info(f"Starting analysis: type='{analysis_type}', file='{filename}'")

    if not llm:
        logger.error("LLM not initialized, cannot perform analysis.")
        return "Error: AI model is not available for analysis.", None

    # --- Step 1: Get Document Text ---
    doc_text: str | None = None
    source_of_text_log = "unknown"
    prefer_nougat_extraction = False

    # Determine if Nougat should be preferred for this analysis type
    if analysis_type == 'mindmap' and NOUGAT_ENABLED and NOUGAT_API_URL:
        prefer_nougat_extraction = True
        logger.info(f"Mindmap analysis for '{filename}'. Nougat extraction will be preferred.")

    # Locate the PDF file
    potential_paths = [
        os.path.join(UPLOAD_FOLDER, filename),
        os.path.join(DEFAULT_PDFS_FOLDER, filename)
    ]
    load_path = next((p for p in potential_paths if os.path.exists(p)), None)

    if not load_path:
        logger.error(f"Document file '{filename}' not found in default or upload folders for analysis.")
        return f"Error: Document '{filename}' not found.", None

    # Attempt text extraction
    if prefer_nougat_extraction:
        logger.debug(f"Attempting extraction for '{filename}' with Nougat preference.")
        doc_text = extract_text_from_pdf(load_path, prefer_nougat=True)
        if doc_text is not None: # Nougat (or its fallback PyMuPDF) succeeded or gave empty text
            source_of_text_log = "Nougat (preferred, or PyMuPDF fallback if Nougat failed)"
        # If doc_text is None here, it means both Nougat and its PyMuPDF fallback failed critically
    
    if doc_text is None: # If Nougat wasn't preferred, or critical failure above
        logger.debug(f"Nougat not preferred or failed. Trying cache/PyMuPDF for '{filename}'.")
        doc_text = document_texts_cache.get(filename) # Try PyMuPDF cache
        if doc_text is not None:
            source_of_text_log = "cache (PyMuPDF)"
        else: # Not in cache, try live PyMuPDF extraction
            logger.debug(f"Text for '{filename}' not in cache. Attempting PyMuPDF live extraction.")
            doc_text = extract_text_from_pdf(load_path, prefer_nougat=False) # Explicitly PyMuPDF
            if doc_text is not None: # PyMuPDF succeeded or gave empty text
                source_of_text_log = "PyMuPDF (live extraction)"
                if doc_text: # Cache if not empty
                    document_texts_cache[filename] = doc_text
            # If doc_text is still None here, PyMuPDF live extraction also failed critically
    
    # Final check on doc_text
    if doc_text is None:
        logger.error(f"Failed to extract text from '{filename}' using any available method.")
        return f"Error: Could not extract text content from '{filename}'. File might be corrupted, empty, or password-protected.", None
    elif not doc_text: # Successfully extracted, but content is empty string
        logger.warning(f"Extracted text for '{filename}' is empty (source: {source_of_text_log}). Analysis might not be useful.")
        # Allow proceeding with empty text, LLM might indicate the document is empty.

    logger.info(f"Using text for '{filename}' (source: {source_of_text_log}, length: {len(doc_text)}) for '{analysis_type}' analysis.")

    # --- Step 2: Prepare Text for LLM (Truncation) ---
    original_length = len(doc_text)
    if original_length > ANALYSIS_MAX_CONTEXT_LENGTH:
        logger.warning(f"Document '{filename}' text too long ({original_length} chars), truncating to {ANALYSIS_MAX_CONTEXT_LENGTH} for '{analysis_type}' analysis.")
        doc_text_for_llm = doc_text[:ANALYSIS_MAX_CONTEXT_LENGTH]
        doc_text_for_llm += "\n\n... [CONTENT TRUNCATED DUE TO LENGTH LIMIT]"
    else:
        doc_text_for_llm = doc_text
        logger.debug(f"Using full document text ({original_length} chars) for analysis '{analysis_type}'.")

    # --- Step 3: Get Analysis Prompt ---
    prompt_template = ANALYSIS_PROMPTS.get(analysis_type)
    if not prompt_template or not isinstance(prompt_template, PromptTemplate):
        logger.error(f"Invalid or missing analysis prompt template for type: {analysis_type} in config.py")
        return f"Error: Invalid analysis type '{analysis_type}' or misconfigured prompt.", None

    try:
        final_prompt = prompt_template.format(doc_text_for_llm=doc_text_for_llm)
        logger.info(f"Sending analysis prompt to LLM (type: {analysis_type}, file: {filename}, model: {OLLAMA_MODEL})...")
        logger.debug(f"Analysis Prompt (Start):\n{final_prompt[:200]}...")

    except KeyError as e:
        logger.error(f"Error formatting ANALYSIS_PROMPTS[{analysis_type}]: Missing key {e}. Check config.py.")
        return f"Error: Internal prompt configuration issue for {analysis_type}.", None
    except Exception as e:
        logger.error(f"Error creating analysis prompt for {analysis_type}: {e}", exc_info=True)
        return f"Error: Could not prepare the request for the {analysis_type} analysis.", None

    try:
        response_object = llm.invoke(final_prompt)
        full_analysis_response = getattr(response_object, 'content', str(response_object))

        logger.info(f"LLM analysis response received for '{filename}' ({analysis_type}). Length: {len(full_analysis_response)}")
        logger.debug(f"Analysis Raw Response (Start):\n{full_analysis_response[:200]}...")

        analysis_content, thinking_content = parse_llm_response(full_analysis_response) 

        if thinking_content:
            logger.info(f"Parsed thinking content from analysis response for '{filename}'.")

        if not analysis_content and thinking_content:
            logger.warning(f"Parsed analysis content is empty for '{filename}' ({analysis_type}). Response only contained thinking.")
            analysis_content = "[Analysis consisted only of reasoning. No final output provided. See thinking process.]"
        elif not analysis_content and not thinking_content:
            logger.error(f"LLM analysis response parsing resulted in empty content and no thinking for '{filename}' ({analysis_type}).")
            analysis_content = "[Analysis generation resulted in empty content after parsing.]"

        logger.info(f"Analysis successful for '{filename}' ({analysis_type}).")
        return analysis_content.strip(), thinking_content
    except Exception as e:
        logger.error(f"LLM analysis invocation error for {filename} ({analysis_type}): {e}", exc_info=True)
        return f"Error generating analysis: AI model failed ({type(e).__name__}). Check logs for details.", None

# --- END OF FILE ai_core.py ---