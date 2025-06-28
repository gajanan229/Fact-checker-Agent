import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Check, Loader2 } from 'lucide-react';

const analysisSteps = [
  { id: 1, text: 'Ingesting Content...', duration: 1500 },
  { id: 2, text: 'Analyzing Transcript & On-Screen Text...', duration: 1500 },
  { id: 3, text: 'Identifying Verifiable Claims...', duration: 1500 },
  { id: 4, text: 'Gathering Evidence from Trusted Sources...', duration: 1500 }
];

const AnalysisStatus: React.FC = () => {
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const processSteps = async () => {
      for (let i = 0; i < analysisSteps.length; i++) {
        setCurrentStep(i);
        await new Promise(resolve => setTimeout(resolve, analysisSteps[i].duration));
        setCompletedSteps(prev => [...prev, analysisSteps[i].id]);
      }
    };

    processSteps();
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto"
    >
      <div className="veritas-card rounded-xl p-8">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-teal-400 to-teal-600 rounded-xl flex items-center justify-center mx-auto mb-4">
            <Loader2 className="w-8 h-8 text-white animate-spin" />
          </div>
          <h2 className="text-2xl font-bold text-gray-100 mb-2">Analysis in Progress</h2>
          <p className="text-gray-400">Our AI agents are investigating the content</p>
        </div>

        <div className="space-y-4">
          {analysisSteps.map((step, index) => {
            const isCompleted = completedSteps.includes(step.id);
            const isCurrent = currentStep === index && !isCompleted;

            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center space-x-3"
              >
                <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                  isCompleted 
                    ? 'bg-teal-500' 
                    : isCurrent 
                    ? 'bg-teal-400 bg-opacity-30 border-2 border-teal-400' 
                    : 'bg-gray-600'
                }`}>
                  {isCompleted ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="progress-checkmark"
                    >
                      <Check className="w-4 h-4 text-white" />
                    </motion.div>
                  ) : isCurrent ? (
                    <Loader2 className="w-3 h-3 text-teal-400 animate-spin" />
                  ) : null}
                </div>
                <span className={`${
                  isCompleted 
                    ? 'text-gray-300' 
                    : isCurrent 
                    ? 'text-teal-400 font-medium' 
                    : 'text-gray-500'
                }`}>
                  {step.text}
                </span>
              </motion.div>
            );
          })}
        </div>

        <div className="mt-8 bg-gray-800 bg-opacity-50 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-sm text-gray-400">
            <div className="w-2 h-2 bg-teal-400 rounded-full animate-pulse"></div>
            <span>Processing... This may take a few moments</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default AnalysisStatus;