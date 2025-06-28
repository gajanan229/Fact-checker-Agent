import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import URLInputForm from '../components/workbench/URLInputForm';
import TargetSelector from '../components/workbench/TargetSelector';
import AnalysisStatus from '../components/workbench/AnalysisStatus';
import CaseFileDashboard from '../components/workbench/CaseFileDashboard';

export const mockCaseFileData = {
  processed_content: {
    transcript: "In today's video, we're going to explore a health myth. Some people claim that drinking a gallon of celery juice every day can reverse aging. We're going to see if that's true.",
    ocr_text: "SHOCKING TRUTH: Celery Juice REVERSES AGING!",
    selected_comment_text: null
  },
  claims: [
    {
      claim_id: 1,
      text: "Drinking a gallon of celery juice every day can reverse aging.",
      status: "verified"
    },
    {
      claim_id: 2,
      text: "The video also mentions that the earth is banana-shaped as a joke.",
      status: "verified"
    }
  ],
  dossier: {
    "1": {
      verdict: "False",
      summary: "While celery juice contains vitamins and antioxidants, there is no scientific evidence to support the claim that it can reverse the biological process of aging. This is a common piece of health misinformation.",
      sources: [
        { id: 1, url: "https://www.healthline.com/nutrition/celery-juice-benefits", title: "Healthline: Does Celery Juice Have Benefits?" },
        { id: 2, url: "https://www.medicalnewstoday.com/articles/325759", title: "Medical News Today: What to know about celery juice" }
      ]
    },
    "2": {
      verdict: "Lacks Context",
      summary: "The statement 'the earth is banana-shaped' was identified, but appears to be used sarcastically or as a joke within the video's context and is not presented as a serious factual claim.",
      sources: []
    }
  },
  draft_response: "Regarding the video's claim about celery juice and aging: \n\nBased on current scientific understanding, the claim that drinking celery juice can reverse aging is false. While celery juice is hydrating and contains nutrients, there is no clinical evidence to support such a significant health claim [1]. Health experts generally advise a balanced diet over relying on a single 'superfood' for dramatic results [2].\n\nSources:\n[1] Healthline: Does Celery Juice Have Benefits? (healthline.com)\n[2] Medical News Today: What to know about celery juice (medicalnewstoday.com)",
  critique: {
    is_revision_needed: true,
    feedback_text: "Initial draft was slightly too clinical. Revised to be more direct and accessible for a general audience."
  }
};

type AppStatus = 'idle' | 'submitting' | 'analyzing' | 'results_displayed' | 'refining' | 'error';

interface Target {
  type: 'video' | 'comment' | null;
  content: string;
}

interface WorkbenchState {
  appStatus: AppStatus;
  videoUrl: string;
  target: Target;
  caseFile: typeof mockCaseFileData | null;
  errorMessage: string;
  showToast: boolean;
  toastMessage: string;
}

const WorkbenchPage: React.FC = () => {
  const [state, setState] = useState<WorkbenchState>({
    appStatus: 'idle',
    videoUrl: '',
    target: { type: null, content: '' },
    caseFile: null,
    errorMessage: '',
    showToast: false,
    toastMessage: ''
  });

  const showToast = (message: string) => {
    setState(prev => ({ ...prev, showToast: true, toastMessage: message }));
    setTimeout(() => {
      setState(prev => ({ ...prev, showToast: false, toastMessage: '' }));
    }, 3000);
  };

  const handleUrlSubmit = (url: string) => {
    setState(prev => ({ ...prev, videoUrl: url, appStatus: 'submitting' }));
    
    // Simulate URL validation
    setTimeout(() => {
      setState(prev => ({ ...prev, appStatus: 'idle' }));
    }, 1000);
  };

  const handleTargetSelect = (type: 'video' | 'comment', content: string = '') => {
    setState(prev => ({ 
      ...prev, 
      target: { type, content },
      appStatus: 'analyzing'
    }));

    // Simulate analysis completion
    setTimeout(() => {
      setState(prev => ({ 
        ...prev, 
        appStatus: 'results_displayed',
        caseFile: mockCaseFileData
      }));
    }, 6000);
  };

  const handleRefinement = (message: string) => {
    setState(prev => ({ ...prev, appStatus: 'refining' }));

    // Simulate AI refinement
    setTimeout(() => {
      if (state.caseFile) {
        const updatedCaseFile = {
          ...state.caseFile,
          draft_response: state.caseFile.draft_response.replace(
            'Based on current scientific understanding',
            'Let me be direct'
          )
        };
        
        setState(prev => ({ 
          ...prev, 
          appStatus: 'results_displayed',
          caseFile: updatedCaseFile
        }));
      }
    }, 2000);
  };

  const handleCopyToClipboard = () => {
    if (state.caseFile?.draft_response) {
      navigator.clipboard.writeText(state.caseFile.draft_response);
      showToast('Comment copied to clipboard!');
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-gray-100 mb-2">Analyst's Workbench</h1>
          <p className="text-gray-400">Investigate claims and craft evidence-based responses</p>
        </motion.header>

        <AnimatePresence mode="wait">
          {state.appStatus === 'idle' && (
            <motion.div
              key="url-input"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <URLInputForm 
                onSubmit={handleUrlSubmit}
                isSubmitting={state.appStatus === 'submitting'}
              />
              {state.videoUrl && (
                <TargetSelector onSelect={handleTargetSelect} />
              )}
            </motion.div>
          )}

          {state.appStatus === 'analyzing' && (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <AnalysisStatus />
            </motion.div>
          )}

          {(state.appStatus === 'results_displayed' || state.appStatus === 'refining') && state.caseFile && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <CaseFileDashboard 
                caseFile={state.caseFile}
                videoUrl={state.videoUrl}
                isRefining={state.appStatus === 'refining'}
                onRefinement={handleRefinement}
                onCopyToClipboard={handleCopyToClipboard}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Toast Notification */}
        <AnimatePresence>
          {state.showToast && (
            <motion.div
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 100 }}
              className="fixed bottom-6 right-6 bg-teal-600 text-white px-6 py-3 rounded-lg shadow-lg toast z-50"
            >
              {state.toastMessage}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default WorkbenchPage;