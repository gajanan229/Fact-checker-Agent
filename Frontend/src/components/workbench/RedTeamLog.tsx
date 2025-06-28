import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, EyeOff, Brain } from 'lucide-react';

interface RedTeamLogProps {
  critique: {
    is_revision_needed: boolean;
    feedback_text: string;
  };
}

const RedTeamLog: React.FC<RedTeamLogProps> = ({ critique }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="veritas-card rounded-xl p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-gray-100">Red Team Log</h3>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center space-x-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
        >
          {isExpanded ? (
            <>
              <EyeOff className="w-4 h-4" />
              <span>Hide Process</span>
            </>
          ) : (
            <>
              <Eye className="w-4 h-4" />
              <span>Show Process</span>
            </>
          )}
        </button>
      </div>

      <p className="text-sm text-gray-400 mb-4">
        Internal quality assurance and bias detection process
      </p>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="bg-purple-500 bg-opacity-10 border border-purple-500 border-opacity-30 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <h4 className="text-sm font-medium text-purple-300 mb-2">
                    Adversarial Review
                  </h4>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    {critique.feedback_text}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!isExpanded && (
        <div className="text-center">
          <div className="inline-flex items-center space-x-2 px-3 py-2 bg-purple-500 bg-opacity-20 border border-purple-500 border-opacity-30 rounded-full">
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-purple-300">Quality checks passed</span>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default RedTeamLog;