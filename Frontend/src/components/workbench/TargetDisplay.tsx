import React from 'react';
import { motion } from 'framer-motion';
import { Play, ExternalLink } from 'lucide-react';

interface TargetDisplayProps {
  videoUrl: string;
}

const TargetDisplay: React.FC<TargetDisplayProps> = ({ videoUrl }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="veritas-card rounded-xl p-6"
    >
      <h3 className="text-lg font-semibold text-gray-100 mb-4">Investigation Target</h3>
      
      <div className="flex items-center space-x-4">
        {/* Video Placeholder */}
        <div className="w-24 h-24 bg-gray-800 rounded-lg flex items-center justify-center border-2 border-gray-700">
          <Play className="w-8 h-8 text-gray-400" />
        </div>

        {/* Video Info */}
        <div className="flex-1">
          <h4 className="font-medium text-gray-100 mb-1">TikTok Video Analysis</h4>
          <p className="text-sm text-gray-400 mb-2">Content: Health claims about celery juice</p>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">{videoUrl}</span>
            <ExternalLink className="w-3 h-3 text-gray-500" />
          </div>
        </div>

        {/* Status Badge */}
        <div className="px-3 py-1 bg-teal-500 bg-opacity-20 border border-teal-500 rounded-full">
          <span className="text-xs font-medium text-teal-400">Analyzed</span>
        </div>
      </div>
    </motion.div>
  );
};

export default TargetDisplay;