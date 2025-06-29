import React from 'react';
import { motion } from 'framer-motion';
import { Send, MessageSquare } from 'lucide-react';

interface CollaborationHubProps {
  onMessage: (message: string) => void;
  isRefining: boolean;
}

const CollaborationHub: React.FC<CollaborationHubProps> = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="verifyp-card rounded-xl p-6 opacity-50"
    >
      <div className="flex items-center space-x-3 mb-4">
        <MessageSquare className="w-5 h-5 text-gray-500" />
        <h3 className="text-lg font-semibold text-gray-400">Collaboration Hub</h3>
        <span className="px-2 py-1 text-xs font-medium text-yellow-300 bg-yellow-500 bg-opacity-20 rounded-full">
          Coming Soon
        </span>
      </div>

      <p className="text-sm text-gray-500 mb-4">
        This feature to refine your response with AI is under development.
      </p>

      {/* Disabled Input Form */}
      <div className="flex space-x-3">
        <input
          type="text"
          placeholder="Refinement features are coming soon..."
          className="flex-1 px-4 py-2 verifyp-input text-gray-100 rounded-lg focus:outline-none"
          disabled={true}
        />
        <button
          type="submit"
          disabled={true}
          className="verifyp-button text-white p-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </motion.div>
  );
};

export default CollaborationHub;