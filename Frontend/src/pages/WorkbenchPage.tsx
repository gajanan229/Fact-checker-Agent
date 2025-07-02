import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import URLInputForm from '../components/workbench/URLInputForm';
import AnalysisStatus from '../components/workbench/AnalysisStatus';
import CaseFileDashboard from '../components/workbench/CaseFileDashboard';

// TypeScript Interfaces for API data
interface Source {
  id: number;
  url: string;
  title: string;
}

interface DossierEntry {
  verdict: string;
  summary: string;
  sources: Source[];
}

interface Claim {
  claim_id: string; // Changed to string to match backend transform
  text: string;
  status: string;
}

interface Critique {
  is_revision_needed: boolean;
  feedback_text: string;
}

interface CaseFile {
  claims: Claim[];
  dossier: { [key: string]: DossierEntry };
  critique: Critique | Record<string, never>;
  draft_response: string;
  final_response: string;
  error_message?: string;
}

// Analysis progress steps
interface ProgressStep {
  id: string;
  text: string;
  status: 'pending' | 'in_progress' | 'completed';
}

const initialProgressSteps: ProgressStep[] = [
  { id: 'ingest_content', text: 'Ingesting Content...', status: 'pending' },
  { id: 'identify_claims', text: 'Identifying Verifiable Claims...', status: 'pending' },
  { id: 'research_claims', text: 'Gathering Evidence...', status: 'pending' },
  { id: 'generate_response', text: 'Drafting Response...', status: 'pending' },
  { id: 'critique_response', text: 'Performing Quality Review...', status: 'pending' },
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type AppStatus = 'idle' | 'submitting' | 'analyzing' | 'results_displayed' | 'refining' | 'error';

interface ApiLimits {
  apify_limit_reached: boolean;
  tavily_limit_reached: boolean;
  gemini_daily_limit_reached: boolean;
}

interface WorkbenchState {
  appStatus: AppStatus;
  videoUrl: string;
  sessionId: string | null;
  caseFile: CaseFile | null;
  progressSteps: ProgressStep[];
  errorMessage: string;
  apiLimits: ApiLimits | null;
}

const WorkbenchPage: React.FC = () => {
  const navigate = useNavigate();
  const [state, setState] = useState<WorkbenchState>({
    appStatus: 'idle',
    videoUrl: '',
    sessionId: null,
    caseFile: null,
    progressSteps: initialProgressSteps,
    errorMessage: '',
    apiLimits: null,
  });

  // Effect to fetch API limits
  useEffect(() => {
    const fetchApiLimits = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/limits/status`);
        if (response.ok) {
          const data = await response.json();
          setState(prev => ({ ...prev, apiLimits: data }));
        }
      } catch (error) {
        console.error("Failed to fetch API limits:", error);
      }
    };

    fetchApiLimits();
  }, []);

  // Effect to manage the EventSource connection for status updates
  useEffect(() => {
    if (state.appStatus !== 'analyzing' || !state.sessionId) {
      return;
    }

    const eventSource = new EventSource(`${API_BASE_URL}/api/status/${state.sessionId}`);

    eventSource.onopen = () => {
      console.log("Connection to event stream opened.");
    };

    // Listener for 'update' events
    eventSource.addEventListener('update', (event) => {
      const eventData = JSON.parse(event.data);
      const { step, status } = eventData;

      if (status === 'completed') {
        setState(prev => ({
          ...prev,
          progressSteps: prev.progressSteps.map(p =>
            p.id === step ? { ...p, status: 'completed' } : p
          )
        }));
      }
    });

    // Listener for 'complete' event
    eventSource.addEventListener('complete', (event) => {
      const eventData = JSON.parse(event.data);
      const caseFile = eventData.caseFile as CaseFile;

      if (caseFile && caseFile.error_message) {
        setState(prev => ({
          ...prev,
          appStatus: 'error',
          errorMessage: caseFile.error_message || 'An unknown error occurred.',
          caseFile: null
        }));
      } else if (caseFile && caseFile.draft_response) {
        setState(prev => ({
          ...prev,
          appStatus: 'results_displayed',
          caseFile: caseFile
        }));
      } else {
        setState(prev => ({
            ...prev,
            appStatus: 'error',
            errorMessage: 'Could not generate a response. This may happen if no verifiable claims were found or if an internal system error occurred.',
            caseFile: null
        }));
      }
      eventSource.close();
    });

    // Listener for 'error' events
    eventSource.addEventListener('error', (event: Event) => {
      // The 'error' event for EventSource doesn't typically contain a data payload.
      // It's usually a sign of a connection failure or the server closing the stream.
      console.error('An error occurred with the event stream:', event);
      
      let message = 'The connection to the server was lost or the analysis failed.';
      // We only try to parse data if it seems to exist.
      const eventWithData = event as MessageEvent;
      if (eventWithData && eventWithData.data) {
          try {
              const eventData = JSON.parse(eventWithData.data);
              if (eventData?.message) {
                  message = eventData.message;
              }
          } catch (e) {
              console.error("Could not parse error event data:", e);
          }
      }

      // Avoid setting an error state if the stream just ended successfully.
      setState(prev => {
        if (prev.appStatus === 'results_displayed') {
          return prev;
        }
        return {
          ...prev,
          appStatus: 'error',
          errorMessage: message
        };
      });

      eventSource.close();
    });

    // Cleanup on component unmount or state change
    return () => {
      console.log("Closing event stream connection.");
      eventSource.close();
    };
  }, [state.appStatus, state.sessionId]);

  const isLimitReached = state.apiLimits ? 
    state.apiLimits.apify_limit_reached || 
    state.apiLimits.tavily_limit_reached || 
    state.apiLimits.gemini_daily_limit_reached 
    : false;

  const handleUrlSubmit = async (url: string) => {
    if (isLimitReached) return;
    setState(prev => ({ ...prev, videoUrl: url, appStatus: 'submitting' }));

    try {
      // Since we are ignoring the comment analysis, we immediately start analysis
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: url })
      });

      if (!response.ok) {
        throw new Error('Failed to start analysis.');
      }

      const data = await response.json();
      setState(prev => ({
        ...prev,
        sessionId: data.session_id,
        appStatus: 'analyzing',
        progressSteps: initialProgressSteps.map(p => 
          p.id === 'ingest_content' ? {...p, status: 'in_progress'} : p
        )
      }));

    } catch (error) {
      console.error(error);
      setState(prev => ({ 
        ...prev, 
        appStatus: 'error', 
        errorMessage: 'Could not connect to the backend server.' 
      }));
    }
  };

  // This function is no longer needed as we start analysis directly
  // const handleTargetSelect = (type: 'video' | 'comment', content: string = '') => { ... }
  
  // Refinement and clipboard logic remain the same for now
  const handleRefinement = (message: string) => {
    // This would be a new API call in a real application
    console.log("Refinement message:", message);
  };

  const handleCopyToClipboard = () => {
    if (state.caseFile?.draft_response) {
      navigator.clipboard.writeText(state.caseFile.draft_response);
      // Toast logic can be re-added if needed
    }
  };

  const renderContent = () => {
    switch (state.appStatus) {
      case 'idle':
      case 'submitting':
        return (
          <div className="relative">
            <URLInputForm 
              onSubmit={handleUrlSubmit}
              isSubmitting={state.appStatus === 'submitting'}
            />
            {isLimitReached && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 bg-gray-900 bg-opacity-80 flex flex-col items-center justify-center rounded-xl"
              >
                <AlertTriangle className="w-10 h-10 text-yellow-400 mb-4" />
                <h3 className="text-xl font-bold text-gray-100 mb-2">Service Temporarily Unavailable</h3>
                <p className="text-yellow-300">
                  Daily usage limits have been reached. Please try again tomorrow.
                </p>
              </motion.div>
            )}
          </div>
        );
      case 'analyzing':
        return <AnalysisStatus steps={state.progressSteps} />;
      case 'results_displayed':
      case 'refining':
        if (state.caseFile) {
          return (
            <CaseFileDashboard 
              caseFile={state.caseFile}
              videoUrl={state.videoUrl}
              isRefining={state.appStatus === 'refining'}
              onRefinement={handleRefinement}
              onCopyToClipboard={handleCopyToClipboard}
            />
          );
        }
        return null; // Or some fallback UI
      case 'error':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center verifyp-card rounded-xl p-8 max-w-2xl mx-auto"
          >
            <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-100 mb-2">Analysis Complete</h2>
            <p className="text-red-300 mb-6">{state.errorMessage}</p>
            <motion.button
              onClick={() => navigate('/')}
              className="verifyp-button text-white font-semibold py-3 px-6 rounded-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Try Another Video
            </motion.button>
          </motion.div>
        );
      default:
        return null;
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
          <motion.div
            key={state.appStatus}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default WorkbenchPage;