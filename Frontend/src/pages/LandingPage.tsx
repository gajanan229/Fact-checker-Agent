import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Shield, Search, Target, Users } from 'lucide-react';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Search className="w-8 h-8" />,
      title: "Deep Analysis",
      description: "AI-powered investigation of video content and comments with comprehensive fact-checking"
    },
    {
      icon: <Target className="w-8 h-8" />,
      title: "Targeted Verification",
      description: "Choose to analyze video content or specific comments with precision targeting"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Transparent Process",
      description: "See every step of the investigation with full transparency and source verification"
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "Collaborative Refinement",
      description: "Work with AI to craft the perfect response that's constructive and well-sourced"
    }
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="px-6 py-4"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-teal-400 to-teal-600 rounded-lg flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-gray-100">Veritas AI</span>
          </div>
        </div>
      </motion.header>

      {/* Hero Section */}
      <main className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="max-w-4xl mx-auto text-center">
          <motion.h1 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl md:text-6xl font-bold text-gray-100 mb-6 leading-tight"
          >
            Challenge Misinformation 
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-300"> with Clarity</span>
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-xl text-gray-300 mb-12 leading-relaxed max-w-3xl mx-auto"
          >
            Veritas AI is your intelligent fact-checking partner for TikTok content. Investigate claims, verify information with credible sources, and craft well-reasoned responses to combat misinformation constructively.
          </motion.p>

          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            onClick={() => navigate('/workbench')}
            className="veritas-button text-white font-semibold py-4 px-8 rounded-lg text-lg shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Start an Investigation
          </motion.button>
        </div>
      </main>

      {/* Features Section */}
      <section className="px-6 py-16">
        <div className="max-w-6xl mx-auto">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="text-3xl font-bold text-gray-100 text-center mb-12"
          >
            How Veritas AI Works
          </motion.h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 1 + index * 0.1 }}
                className="veritas-card rounded-xl p-6 text-center"
              >
                <div className="text-teal-400 mb-4 flex justify-center">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-100 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 border-t border-gray-700">
        <div className="max-w-7xl mx-auto text-center text-gray-400">
          <p>&copy; 2025 Veritas AI. Empowering truth through technology.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;