AI HR Policy Assistant 🤖

📌 Project Overview

The AI HR Policy Assistant is a chatbot that answers employee questions using HR policy documents. It uses Retrieval-Augmented Generation (RAG) to provide relevant answers based on the available documents.

🎯 Features

- 📄 HR policy document ingestion
- 🔍 Semantic search using FAISS vector database
- 💬 AI-powered HR policy question answering
- 📚 Source references for answers
- 👩‍💼 Escalation to HR when information is unavailable
- 📝 Answer logging

🛠️ Technologies Used

- Python
- Streamlit
- FAISS
- Hugging Face API
- Hugging Face Embeddings
- Retrieval-Augmented Generation (RAG)

📂 Project Structure

- "app.py" – Main application
- "requirements.txt" – Project dependencies
- "HR_Policy.pdf" – HR policy document (if included)

🚀 How to Run the Project

1. Clone this repository:
   
   git clone YOUR_GITHUB_REPOSITORY_URL

2. Open the project folder:
   
   cd YOUR_PROJECT_FOLDER

3. Install dependencies:
   
   pip install -r requirements.txt

4. Create a ".env" file and add your Hugging Face API key:
   
   HF_TOKEN=your_huggingface_api_key

5. Run the application:
   
   streamlit run app.py

🔐 Security

Never upload your API keys, passwords, or ".env" file to GitHub.

👩‍💻 Author

Bharathi

📌 Project Type

AI / Machine Learning / Generative AI / RAG Chatbot
