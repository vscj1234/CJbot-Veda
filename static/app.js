class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            micButton: document.querySelector('#mic-button'),
            inputField: document.querySelector('.chatbox__footer input')
        }

        this.state = false;
        this.messages = [];
        this.inactivityTimer = null;
        this.inactivityPromptDisplayed = false;

        // Initialize Speech Recognition
        this.recognition = null;
        this.initSpeechRecognition();

        this.startInactivityTimer();
    }

    initSpeechRecognition() {
        // Check for browser support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.lang = 'en-US';
            this.recognition.interimResults = false;
            this.recognition.maxAlternatives = 1;

            // Handle the result event
            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.args.inputField.value = transcript;
                this.onSendButton(this.args.chatBox);
            }

            // Handle errors
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
            }
        } else {
            console.warn('Speech Recognition API not supported in this browser.');
        }
    }

    display() {
        const {openButton, chatBox, sendButton, micButton, inputField} = this.args;
        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        micButton.addEventListener('click', () => this.onMicButton());
        inputField.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
        chatBox.addEventListener('click', () => this.resetInactivityTimer());
        inputField.addEventListener('click', () => this.resetInactivityTimer());
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // Show or hide the chatbox
        if(this.state) {
            chatbox.classList.add('chatbox--active');
            this.resetInactivityTimer();
        } else {
            chatbox.classList.remove('chatbox--active');
            this.clearInactivityTimer();
        }
    }

    onMicButton() {
        if (this.recognition) {
            if (this.recognitionActive) {
                this.recognition.stop();
                this.recognitionActive = false;
            } else {
                this.recognition.start();
                this.recognitionActive = true;
            }
        }
    }

    onSendButton(chatbox) {
        var textField = this.args.inputField;
        let text1 = textField.value.trim();
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        let typingMessage = { name: "CloudJune", message: "Typing..." };
        this.messages.push(typingMessage);
        this.updateChatText(chatbox);

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            // Remove the "typing" message
            this.messages.pop();
            let msg2 = { name: "CloudJune", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            this.speak(r.answer); // Speak the response
            textField.value = '';
            this.resetInactivityTimer();
        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            textField.value = '';
            this.resetInactivityTimer();
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "CloudJune") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }

    speak(message) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        } else {
            console.warn('Speech Synthesis API not supported in this browser.');
        }
    }

    startInactivityTimer() {
        this.inactivityTimer = setInterval(() => {
            if (!this.inactivityPromptDisplayed) {
                this.showInactivityPrompt();
            }
        }, 180000); // 3 minutes
    }

    resetInactivityTimer() {
        this.clearInactivityTimer();
        this.inactivityPromptDisplayed = false;
        this.startInactivityTimer();
    }

    clearInactivityTimer() {
        clearInterval(this.inactivityTimer);
    }

    showInactivityPrompt() {
        this.inactivityPromptDisplayed = true;
        if (confirm("Do you want to continue the chat?")) {
            this.resetInactivityTimer();
        } else {
            this.terminateSession();
        }
    }

    terminateSession() {
        this.clearInactivityTimer();
        const chatbox = this.args.chatBox;
        let msg = { name: "CloudJune", message: "Bye! Have a great day." };
        this.messages.push(msg);
        this.updateChatText(chatbox);
        this.speak(msg.message); // Speak the goodbye message
        chatbox.classList.remove('chatbox--active');
    }
}

const chatbox = new Chatbox();
chatbox.display();
