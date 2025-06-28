import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Loader2 } from 'lucide-react';

interface DraftResponseProps {
  response: string;
  isRefining: boolean;
}

const DraftResponse: React.FC<DraftResponseProps> = ({ response, isRefining }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="veritas-card rounded-xl p-6"
    >
      <div className="flex items-center space-x-3 mb-4">
        <FileText className="w-5 h-5 text-teal-400" />
        <h3 className="text-lg font-semibold text-gray-100">Draft Response</h3>
        {isRefining && (
          <div className="flex items-center space-x-2 text-sm text-teal-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Refining...</span>
          </div>
        )}
      </div>

      <div className="bg-gray-800 bg-opacity-50 rounded-lg p-4 border border-gray-700">
        <div className="text-gray-200 leading-relaxed whitespace-pre-line">
          {response}
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-400">
        <p>This response has been fact-checked and is ready for posting. You can refine it further using the collaboration tools below.</p>
      </div>
    </motion.div>
  );
};

export default DraftResponse;