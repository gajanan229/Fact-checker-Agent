import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, Loader2 } from 'lucide-react';

interface URLInputFormProps {
  onSubmit: (url: string) => void;
  isSubmitting: boolean;
}

const URLInputForm: React.FC<URLInputFormProps> = ({ onSubmit, isSubmitting }) => {
  const [url, setUrl] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onSubmit(url.trim());
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto"
    >
      <div className="verifyp-card rounded-xl p-8">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-teal-400 to-teal-600 rounded-xl flex items-center justify-center mx-auto mb-4">
            <Link className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-100 mb-2">Start Your Investigation</h2>
          <p className="text-gray-400">Submit a TikTok link to begin analyzing content or comments</p>
          <p className="text-sm text-yellow-400 mt-2">
            ⚠️ Please ensure the TikTok video has clear audio and verifiable claims for accurate analysis.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="url" className="block text-sm font-medium text-gray-300 mb-2">
              TikTok URL
            </label>
            <input
              type="url"
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.tiktok.com/@username/video/..."
              className="w-full px-4 py-3 verifyp-input text-gray-100 rounded-lg focus:outline-none"
              disabled={isSubmitting}
              required
            />
          </div>

          <motion.button
            type="submit"
            disabled={isSubmitting || !url.trim()}
            className="w-full verifyp-button text-white font-semibold py-3 px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            whileHover={!isSubmitting ? { scale: 1.02 } : {}}
            whileTap={!isSubmitting ? { scale: 0.98 } : {}}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="animate-spin w-5 h-5 mr-2" />
                Validating URL...
              </>
            ) : (
              'Analyze Content'
            )}
          </motion.button>
        </form>
      </div>
    </motion.div>
  );
};

export default URLInputForm;