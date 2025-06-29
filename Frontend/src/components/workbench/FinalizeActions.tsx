import React from 'react';
import { motion } from 'framer-motion';
import { Copy, Download, Share } from 'lucide-react';

interface FinalizeActionsProps {
  onCopyToClipboard: () => void;
}

const FinalizeActions: React.FC<FinalizeActionsProps> = ({ onCopyToClipboard }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="verifyp-card rounded-xl p-6"
    >
      <h3 className="text-lg font-semibold text-gray-100 mb-4">Finalize & Deploy</h3>
      
      <p className="text-sm text-gray-400 mb-6">
        Your fact-checked response is ready. Copy it to your clipboard and post it on TikTok to combat misinformation.
      </p>

      <div className="flex flex-col sm:flex-row gap-3">
        <motion.button
          onClick={onCopyToClipboard}
          className="flex-1 verifyp-button text-white font-semibold py-3 px-6 rounded-lg flex items-center justify-center space-x-2"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Copy className="w-5 h-5" />
          <span>Copy Final Comment</span>
        </motion.button>

        <motion.button
          className="verifyp-secondary-button font-medium py-3 px-6 rounded-lg flex items-center justify-center space-x-2"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Download className="w-5 h-5" />
          <span>Save Report</span>
        </motion.button>

        <motion.button
          className="verifyp-secondary-button font-medium py-3 px-6 rounded-lg flex items-center justify-center space-x-2"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Share className="w-5 h-5" />
          <span>Share Case</span>
        </motion.button>
      </div>

      <div className="mt-4 p-4 bg-teal-500 bg-opacity-10 border border-teal-500 border-opacity-30 rounded-lg">
        <div className="flex items-start space-x-3">
          <div className="w-2 h-2 bg-teal-400 rounded-full mt-2 flex-shrink-0"></div>
          <div>
            <h4 className="text-sm font-medium text-teal-300 mb-1">
              Remember: Constructive Engagement
            </h4>
            <p className="text-xs text-gray-300">
              Post your response respectfully to promote healthy discussion and fact-based dialogue.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default FinalizeActions;