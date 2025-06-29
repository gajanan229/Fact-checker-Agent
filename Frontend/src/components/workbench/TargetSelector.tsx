import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Video, MessageCircle, X } from 'lucide-react';

interface TargetSelectorProps {
  onSelect: (type: 'video' | 'comment', content?: string) => void;
}

const TargetSelector: React.FC<TargetSelectorProps> = ({ onSelect }) => {
  const [selectedType, setSelectedType] = useState<'video' | 'comment' | null>(null);
  const [username, setUsername] = useState('');
  const [showModal, setShowModal] = useState(true);

  const handleConfirm = () => {
    if (selectedType === 'video') {
      onSelect('video');
    } else if (selectedType === 'comment' && username.trim()) {
      onSelect('comment', username.trim());
    }
    setShowModal(false);
  };

  if (!showModal) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        className="verifyp-card rounded-xl p-8 max-w-md w-full"
      >
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold text-gray-100">Select Investigation Target</h3>
          <button
            onClick={() => setShowModal(false)}
            className="text-gray-400 hover:text-gray-200"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <p className="text-gray-400 mb-6">What would you like to investigate?</p>

        <div className="space-y-4 mb-6">
          <motion.button
            onClick={() => setSelectedType('video')}
            className={`w-full p-4 rounded-lg border transition-colors flex items-center space-x-3 ${
              selectedType === 'video'
                ? 'border-teal-400 bg-teal-400 bg-opacity-10'
                : 'border-gray-600 hover:border-teal-400'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Video className="w-6 h-6 text-teal-400" />
            <div className="text-left">
              <div className="font-semibold text-gray-100">Video Content</div>
              <div className="text-sm text-gray-400">Analyze the video's audio and visual text</div>
            </div>
          </motion.button>

          <motion.button
            onClick={() => setSelectedType('comment')}
            className={`w-full p-4 rounded-lg border transition-colors flex items-center space-x-3 ${
              selectedType === 'comment'
                ? 'border-teal-400 bg-teal-400 bg-opacity-10'
                : 'border-gray-600 hover:border-teal-400'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <MessageCircle className="w-6 h-6 text-teal-400" />
            <div className="text-left">
              <div className="font-semibold text-gray-100">Specific Comment</div>
              <div className="text-sm text-gray-400">Investigate claims in a user's comment</div>
            </div>
          </motion.button>
        </div>

        {selectedType === 'comment' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mb-6"
          >
            <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
              Username
            </label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="@username"
              className="w-full px-3 py-2 verifyp-input text-gray-100 rounded-lg focus:outline-none"
            />
          </motion.div>
        )}

        <div className="flex space-x-3">
          <button
            onClick={() => setShowModal(false)}
            className="flex-1 verifyp-secondary-button py-2 px-4 rounded-lg font-medium"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={!selectedType || (selectedType === 'comment' && !username.trim())}
            className="flex-1 verifyp-button text-white py-2 px-4 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Confirm
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default TargetSelector;