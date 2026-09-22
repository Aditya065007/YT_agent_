# YouTube Video RAG Chatbot

**Live App:** https://dhf8vssacsym8vtqth9hgz.streamlit.app/

A Streamlit app that lets you load one or more YouTube videos and ask questions about their content in a conversational, multi-turn chat interface. Answers are grounded strictly in the video transcripts using a technique called retrieval-augmented generation (RAG) — instead of letting a language model answer from general knowledge, the app first finds the exact parts of the transcript relevant to the question and forces the model to answer only from that material.

## Features

- **Multi-video support** — load several videos into a single session; new videos are merged into one shared searchable index rather than replacing the previous one
- **Multi-turn conversation** — follow-up questions are answered with awareness of prior chat history, not treated as isolated queries
- **Grounded answers** — the model is explicitly instructed to answer only from retrieved transcript content and to say "I don't know" when the transcript doesn't cover the question, rather than filling gaps from general knowledge
- **Session management** — track and clear loaded videos from the sidebar

## Tech Stack, and why each piece is here

- **LangChain** — a framework for chaining together the different steps an AI app needs (load data, break it into pieces, search it, send it to a model). Rather than writing all that plumbing by hand, LangChain provides ready-made building blocks for each step.
- **FAISS (Facebook AI Similarity Search)** — a library built specifically for fast similarity search over large numbers of text embeddings. Think of it as a search index, but instead of matching keywords, it matches *meaning* — it can find the transcript chunk that's semantically closest to a question, even if the wording is completely different.
- **HuggingFace `all-MiniLM-L6-v2`** — the embedding model. An embedding model converts a piece of text into a list of numbers (a vector) that represents its meaning. Two pieces of text about the same topic end up with similar vectors, which is what makes searching by meaning possible. This particular model is small and fast, which matters because it has to run on every chunk of every video.
- **Groq (running Llama 3.3 70B)** — the actual language model that reads the retrieved transcript pieces and writes the answer. Groq is a hosting service known for very fast inference speed, which keeps the chatbot feeling responsive.
- **Streamlit** — the web framework used to build the interface itself (the chat window, the sidebar, the buttons) without writing separate frontend code.

## How It Works, step by step

1. **Getting the transcript.** When you paste a YouTube URL, `YoutubeLoader` pulls the video's transcript text directly — no audio processing needed, since YouTube already provides captions/transcripts for most videos.
2. **Breaking it into chunks.** A full video transcript can be thousands of words long, which is too much to search efficiently or fit into a single question to the model. `RecursiveCharacterTextSplitter` breaks the transcript into overlapping chunks of about 1,000 characters each (with 100 characters of overlap between chunks, so a sentence that straddles two chunks doesn't lose context).
3. **Turning chunks into searchable vectors.** Each chunk is passed through the embedding model, converting it into a vector. These vectors are stored in a FAISS index — essentially a specialized database built for finding "which of these thousands of vectors is closest to this new vector."
4. **Merging multiple videos.** If you load a second video, its chunks go through the same process and are merged into the *same* FAISS index (`FAISS.merge_from`), rather than starting a new one — this is what allows the chatbot to answer questions that might draw on more than one video.
5. **Answering a question.** When you ask something, your question is embedded the same way, and FAISS returns the 4 chunks whose vectors are closest to it — i.e., the 4 most relevant pieces of transcript.
6. **Generating the answer.** Those 4 chunks, along with the last several turns of conversation history, are inserted into a prompt template that explicitly tells the model: only use this transcript text, don't guess, say "I don't know" if it's not covered here, and format the answer as numbered points. This prompt is sent to the Groq-hosted model, and the response is shown in the chat window.


```

Paste a YouTube URL in the sidebar, wait for the transcript to load, then ask questions in the chat input.
