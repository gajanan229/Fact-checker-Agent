import React from 'react';
import { motion } from 'framer-motion';
import { Play, ExternalLink } from 'lucide-react';

interface TargetMetadata {
  url: string;
  domain?: string;
  video_id?: string;
  content_type?: string;
  transcript_excerpt?: string;
  claims_count?: number;
}

interface TargetDisplayProps {
  videoUrl: string;
  target?: TargetMetadata;
}

const TargetDisplay: React.FC<TargetDisplayProps> = ({ videoUrl, target }) => {
  const url = target?.url || videoUrl;
  const domain = target?.domain || '';
  const videoId = target?.video_id || '';
  const excerpt = target?.transcript_excerpt || '';
  const claimsCount = target?.claims_count ?? 0;
  const claimsLabel = `${claimsCount} ${claimsCount === 1 ? 'claim' : 'claims'}`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="verifyp-card rounded-xl p-6"
    >
      <h3 className="text-lg font-semibold text-gray-100 mb-4">Investigation Target</h3>

      <div className="flex items-start space-x-4">
        <div className="w-24 h-24 bg-gray-800 rounded-lg flex items-center justify-center border-2 border-gray-700 flex-shrink-0">
          <Play className="w-8 h-8 text-gray-400" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3 mb-2">
            <div className="min-w-0">
              <h4 className="font-medium text-gray-100">TikTok Video Analysis</h4>
              {(domain || videoId) && (
                <p className="text-xs text-gray-500 mt-0.5 truncate">
                  {domain}
                  {domain && videoId ? ' · ' : ''}
                  {videoId ? `ID ${videoId}` : ''}
                </p>
              )}
            </div>
            <div className="px-3 py-1 bg-teal-500 bg-opacity-20 border border-teal-500 rounded-full whitespace-nowrap flex-shrink-0">
              <span className="text-xs font-medium text-teal-400">{claimsLabel}</span>
            </div>
          </div>

          {excerpt ? (
            <p className="text-sm text-gray-300 italic mb-3 line-clamp-3">
              &ldquo;{excerpt}&rdquo;
            </p>
          ) : (
            <p className="text-sm text-gray-500 italic mb-3">
              Transcript preview unavailable.
            </p>
          )}

          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1.5 text-xs text-gray-400 hover:text-teal-300 transition-colors max-w-full"
          >
            <span className="truncate">{url}</span>
            <ExternalLink className="w-3 h-3 flex-shrink-0" />
          </a>
        </div>
      </div>
    </motion.div>
  );
};

export default TargetDisplay;
