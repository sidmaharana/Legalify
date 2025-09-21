/**
 * Legalify Frontend Script
 *
 * This script handles the user interface interactions for the Legalify application,
 * including document upload, communication with the backend API, managing UI state,
 * and displaying conversational chat.
 */

// --- DOM Element Selections ---
// Select various HTML elements by their IDs for easy access and manipulation.
const uploadForm = document.getElementById('upload-container').querySelector('form');
const fileInput = document.getElementById('file-input');
const analyzeButton = document.getElementById('analyze-button');
const loader = document.getElementById('loader');
const chatContainer = document.getElementById('chat-container');
const uploadContainer = document.getElementById('upload-container');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatBox = document.getElementById('chat-box');
const newAnalysisButton = document.getElementById('new-analysis-button');
const glossaryButton = document.getElementById('glossary-button');
const glossaryModal = document.getElementById('glossary-modal');
const glossaryContent = document.getElementById('glossary-content');
const closeModalButton = document.querySelector('.close-button');


// --- State Management ---
// Stores the base64-encoded FAISS index received from the backend after document processing.
// This index represents the document's knowledge base.
let faissIndex = null;
// Stores the full extracted text of the document for use in other API calls (e.g., glossary).
let extractedText = null;
// Stores the ongoing conversation history between the user and the AI.
// This is sent to the backend with each question to maintain context.
let chatHistory = [];

// --- Event Listeners ---
// Attaches a submit event listener to the document upload form.
uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevents the default form submission behavior (page reload).
    await handleAnalysis(); // Calls the asynchronous function to handle document analysis.
});

// Attaches a submit event listener to the chat input form.
chatForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevents the default form submission behavior.
    await handleQuestion(); // Calls the asynchronous function to handle user questions.
});

// Attaches a click event listener to the 'New Analysis' button.
newAnalysisButton.addEventListener('click', () => {
    resetUI();
});

// Attaches a click event listener to the 'Glossary' button.
glossaryButton.addEventListener('click', async () => {
    await handleGlossary();
});

// Attaches a click event listener to the modal's close button.
closeModalButton.addEventListener('click', () => {
    glossaryModal.classList.add('hidden');
});

// --- Helper Functions ---
/**
 * Adds a message to the chat box display.
 * @param {string} text - The message content to display.
 * @param {string} sender - The sender of the message ('user' or 'ai').
 * @param {string} [className=''] - Optional CSS class to apply to the message div for custom styling (e.g., 'error-message').
 */
function addMessageToChat(text, sender, className = '') {
    const messageDiv = document.createElement('div');
    // Add base 'message' class and sender-specific class (e.g., 'user-message', 'ai-message').
    messageDiv.classList.add('message', `${sender}-message`);
    // Add any additional class provided (e.g., 'error-message').
    if (className) {
        messageDiv.classList.add(className);
    }
    // For AI messages, parse Markdown content using marked.js; otherwise, use plain text.
    messageDiv.innerHTML = (sender === 'ai') ? marked.parse(text) : text;
    chatBox.appendChild(messageDiv);
    // Automatically scroll the chat box to the bottom to show the latest message.
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv; // Return the messageDiv so sources can be appended to it
}

/**
 * Resets the application to its initial state.
 */
function resetUI() {
    // Reset state variables
    faissIndex = null;
    extractedText = null;
    chatHistory = [];

    // Clear UI elements
    chatBox.innerHTML = '';
    fileInput.value = '';

    // Toggle visibility of containers
    chatContainer.classList.add('hidden');
    uploadContainer.classList.remove('hidden');
    loader.classList.add('hidden');
}


/**
 * Centralized API client for making fetch requests to the backend.
 * Handles common tasks like constructing URLs, checking response status, and parsing JSON.
 * Throws detailed errors for easier debugging and user feedback.
 * @param {string} endpoint - The API endpoint (e.g., '/api/process').
 * @param {object} options - Standard Fetch API options (e.g., method, headers, body).
 * @returns {Promise<object>} - A promise that resolves with the JSON response data.
 * @throws {Error} - Throws an error if the network request fails or the API returns an error status.
 */
async function apiClient(endpoint, options) {
    const url = `http://127.0.0.1:8080${endpoint}`; // Construct the full URL to the backend.
    const response = await fetch(url, options);

    // Check if the HTTP response status is OK (200-299).
    if (!response.ok) {
        let errorData = {};
        try {
            // Attempt to parse error details from the response body if it's JSON.
            errorData = await response.json();
        } catch (e) {
            // If the response is not JSON, create a generic error message.
            throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`);
        }
        // Throw an error with a message from the API or a generic one.
        throw new Error(errorData.error || `API error! Status: ${response.status}`);
    }

    return response.json(); // Parse and return the JSON response data.
}

// --- API Logic ---
/**
 * Handles the document analysis process.
 * - Retrieves the selected file from the input.
 * - Manages UI state (shows loader, hides upload form).
 * - Sends the file to the backend's /api/process endpoint using apiClient.
 * - Stores the received FAISS index and transitions the UI to the chat interface.
 * - Displays summary and suggested questions.
 * - Implements robust error handling and ensures the loader is hidden.
 */
async function handleAnalysis() {
    const file = fileInput.files[0];

    if (!file) {
        // Display an error message if no file is selected.
        addMessageToChat('Please select a file to analyze.', 'ai', 'error-message');
        return;
    }

    // Show the loading indicator and hide the upload form to provide user feedback.
    loader.classList.remove('hidden');
    uploadContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send the file to the backend for processing.
        const data = await apiClient('/api/process', {
            method: 'POST',
            body: formData, // FormData is automatically set with 'Content-Type': 'multipart/form-data'
        });

        // Store state from the backend response
        faissIndex = data.faiss_index;
        extractedText = data.extracted_text; // Save the full text for the glossary feature

        // Transition the UI from the upload state to the chat interface.
        chatContainer.classList.remove('hidden');
        // Display the generated summary as the first AI message.
        addMessageToChat(data.summary, 'ai');
        addMessageToChat('Document processed. Ready to answer questions!', 'ai');

        // Display suggested questions as clickable buttons if available.
        if (data.suggested_questions && data.suggested_questions.length > 0) {
            const suggestedQuestionsDiv = document.createElement('div');
            suggestedQuestionsDiv.classList.add('suggested-questions');

            data.suggested_questions.forEach(q => {
                const button = document.createElement('button');
                button.classList.add('suggested-q-btn');
                button.textContent = q;
                // Attach a click listener to each button to auto-fill and submit the question.
                button.addEventListener('click', async () => {
                    userInput.value = q; // Set the input field value to the suggested question.
                    await handleQuestion(); // Programmatically submit the question.
                });
                suggestedQuestionsDiv.appendChild(button);
            });
            chatBox.appendChild(suggestedQuestionsDiv);
            chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom to show new buttons.
        }

    } catch (error) {
        // Log the error to the console for debugging.
        console.error('Error during document analysis:', error);
        // Display a user-friendly error message directly in the chat window.
        addMessageToChat(`Error: ${error.message}. Please try again.`, 'ai', 'error-message');
        // Reset the UI to its initial state on error.
        uploadContainer.classList.remove('hidden');
        chatContainer.classList.add('hidden');
    } finally {
        // Ensure the loader is always hidden, regardless of success or failure.
        loader.classList.add('hidden');
    }
}

/**
 * Handles the user's question submission.
 * - Retrieves the question from the input field.
 * - Displays the user's question in the chat box.
 * - Clears the input field.
 * - Shows a temporary 'typing...' indicator.
 * - Calls the backend's /api/ask endpoint using apiClient, including chat history.
 * - Displays the AI's answer or an error message.
 */
async function handleQuestion() {
    const question = userInput.value.trim();

    if (!question) {
        return; // Do nothing if the question input is empty.
    }

    addMessageToChat(question, 'user'); // Display the user's question in the chat.
    userInput.value = ''; // Clear the input field immediately.

    // Display a temporary 'AI is typing...' message.
    const typingIndicator = addMessageToChat('AI is typing...', 'ai');

    try {
        // Send the question and the FAISS index (and chat history) to the backend.
        const data = await apiClient('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                faiss_index: faissIndex,
                chat_history: chatHistory, // Include the current conversation history.
            }),
        });

        // Remove the 'typing...' indicator once the AI response is received.
        chatBox.removeChild(typingIndicator);

        const aiMessageDiv = addMessageToChat(data.answer, 'ai'); // Display the AI's answer.
        chatHistory = data.chat_history; // Update the chat history with the latest from the backend.

        // Display sources if available
        if (data.sources && data.sources.length > 0) {
            const sourceContainer = document.createElement('div');
            sourceContainer.classList.add('source-container');

            data.sources.forEach(source => {
                const sourceTag = document.createElement('span');
                sourceTag.classList.add('source-tag');
                sourceTag.textContent = `Source: ${source.source}`; // Assuming source.source contains the page/document info
                sourceContainer.appendChild(sourceTag);
            });
            aiMessageDiv.appendChild(sourceContainer);
            chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom to show new sources
        }

    } catch (error) {
        console.error('Error during Q&A:', error);
        // Remove the typing indicator if it's still present.
        if (chatBox.contains(typingIndicator)) {
            chatBox.removeChild(typingIndicator);
        }
        // Display a user-friendly error message in the chat window.
        addMessageToChat(`Error: ${error.message}. Please try again.`, 'ai', 'error-message');
    }
}

/**
 * Handles the creation and display of the glossary.
 * - Shows the glossary modal with a loading indicator.
 * - Calls the backend's /api/glossary endpoint.
 * - Renders the returned key-value pairs into a definition list.
 */
async function handleGlossary() {
    if (!extractedText) {
        alert("Please analyze a document first.");
        return;
    }

    // Show the modal and display a loading message.
    glossaryModal.classList.remove('hidden');
    glossaryContent.innerHTML = '<p>Generating glossary...</p>';

    try {
        const glossaryData = await apiClient('/api/glossary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ extracted_text: extractedText }),
        });

        glossaryContent.innerHTML = ''; // Clear loading message

        if (Object.keys(glossaryData).length === 0) {
            glossaryContent.innerHTML = '<p>No legal jargon was identified in this document.</p>';
            return;
        }

        const dl = document.createElement('dl');
        for (const term in glossaryData) {
            const dt = document.createElement('dt');
            dt.textContent = term;
            const dd = document.createElement('dd');
            dd.textContent = glossaryData[term];
            dl.appendChild(dt);
            dl.appendChild(dd);
        }
        glossaryContent.appendChild(dl);

    } catch (error) {
        console.error('Error generating glossary:', error);
        glossaryContent.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
    }
}