import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Loader2, Copy, Check } from 'lucide-react';

interface ResponseSource {
  number: number;
  domain: string;
  title?: string;
  url?: string;
}

interface DraftResponseProps {
  response: string;
  sources?: ResponseSource[];
  isRefining: boolean;
}

const DraftResponse: React.FC<DraftResponseProps> = ({ response, sources = [], isRefining }) => {
  const [copiedResponse, setCopiedResponse] = useState(false);
  const [copiedSources, setCopiedSources] = useState(false);

  const handleCopyResponse = async () => {
    try {
      await navigator.clipboard.writeText(response);
      setCopiedResponse(true);
      setTimeout(() => setCopiedResponse(false), 2000);
    } catch (err) {
      console.error('Failed to copy response:', err);
    }
  };

  const handleCopySources = async () => {
    if (sources.length === 0) return;
    
    const sourcesText = sources
      .map(source => `[${source.number}] ${source.domain}`)
      .join('\n');
    
    try {
      await navigator.clipboard.writeText(sourcesText);
      setCopiedSources(true);
      setTimeout(() => setCopiedSources(false), 2000);
    } catch (err) {
      console.error('Failed to copy sources:', err);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="verifyp-card rounded-xl p-6"
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

      {/* Main Response */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg p-4 border border-gray-700 mb-4">
        <div className="text-gray-200 leading-relaxed whitespace-pre-line">
          {response}
        </div>
        
        <div className="mt-4 flex justify-end">
          <button
            onClick={handleCopyResponse}
            className="flex items-center space-x-2 px-3 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors text-sm"
          >
            {copiedResponse ? (
              <>
                <Check className="w-4 h-4" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                <span>Copy Response</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Sources Section */}
      {sources.length > 0 && (
        <div className="bg-gray-800 bg-opacity-30 rounded-lg p-4 border border-gray-700">
          <h4 className="text-sm font-semibold text-gray-300 mb-3">Sources:</h4>
          <div className="space-y-2">
            {sources.map((source) => (
              <div key={source.number} className="text-sm text-gray-400">
                <span className="text-teal-400 font-medium">[{source.number}]</span>{' '}
                <span>{source.domain}</span>
                {source.title && (
                  <span className="text-gray-500"> - {source.title}</span>
                )}
              </div>
            ))}
          </div>
          
          <div className="mt-4 flex justify-end">
            <button
              onClick={handleCopySources}
              className="flex items-center space-x-2 px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors text-sm"
            >
              {copiedSources ? (
                <>
                  <Check className="w-4 h-4" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  <span>Copy Sources</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      <div className="mt-4 text-sm text-gray-400">
        <p>This response has been fact-checked and is ready for posting. You can refine it further using the collaboration tools below.</p>
      </div>
    </motion.div>
  );
};

export default DraftResponse;