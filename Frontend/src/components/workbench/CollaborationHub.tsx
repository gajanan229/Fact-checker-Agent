import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, MessageSquare, Loader2 } from 'lucide-react';

interface CollaborationHubProps {
  onMessage: (message: string) => void;
  isRefining: boolean;
}

const CollaborationHub: React.FC<CollaborationHubProps> = ({ onMessage, isRefining }) => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<Array<{ type: 'user' | 'ai', text: string }>>([]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isRefining) {
      // Add user message to chat
      setChatHistory(prev => [...prev, { type: 'user', text: message }]);
      
      // Trigger refinement
      onMessage(message);
      
      // Clear input
      setMessage('');
      
      // Add AI response after delay (simulated)
      setTimeout(() => {
        setChatHistory(prev => [...prev, { 
          type: 'ai', 
          text: 'I\'ve updated the response to be more direct and accessible. The tone has been adjusted while maintaining accuracy and source citations.'
        }]);
      }, 2000);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="veritas-card rounded-xl p-6"
    >
      <div className="flex items-center space-x-3 mb-4">
        <MessageSquare className="w-5 h-5 text-teal-400" />
        <h3 className="text-lg font-semibold text-gray-100">Collaboration Hub</h3>
      </div>

      <p className="text-sm text-gray-400 mb-4">
        Work with the AI to refine your response. Ask for tone changes, additional sources, or clarifications.
      </p>

      {/* Chat History */}
      {chatHistory.length > 0 && (
        <div className="mb-4 space-y-3 max-h-48 overflow-y-auto">
          {chatHistory.map((chat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`chat-message ${
                chat.type === 'user' 
                  ? 'ml-8 bg-teal-500 bg-opacity-20 border-l-4 border-teal-400' 
                  : 'mr-8 bg-gray-700 bg-opacity-50 border-l-4 border-gray-500'
              } p-3 rounded-r-lg`}
            >
              <div className={`text-xs font-medium mb-1 ${
                chat.type === 'user' ? 'text-teal-300' : 'text-gray-400'
              }`}>
                {chat.type === 'user' ? 'You' : 'AI Assistant'}
              </div>
              <p className="text-sm text-gray-200">{chat.text}</p>
            </motion.div>
          ))}
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex space-x-3">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={isRefining ? "Processing your request..." : "Make the tone more direct"}
          className="flex-1 px-4 py-2 veritas-input text-gray-100 rounded-lg focus:outline-none"
          disabled={isRefining}
        />
        <button
          type="submit"
          disabled={!message.trim() || isRefining}
          className="veritas-button text-white p-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {isRefining ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>

      {chatHistory.length === 0 && (
        <div className="mt-4 text-center">
          <div className="inline-flex items-center space-x-2 px-3 py-2 bg-gray-700 bg-opacity-30 rounded-full">
            <div className="w-2 h-2 bg-teal-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-gray-400">AI assistant ready to help</span>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default CollaborationHub;