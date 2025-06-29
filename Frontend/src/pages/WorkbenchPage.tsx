import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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

const API_BASE_URL = "http://localhost:8000";

type AppStatus = 'idle' | 'submitting' | 'analyzing' | 'results_displayed' | 'refining' | 'error';

interface WorkbenchState {
  appStatus: AppStatus;
  videoUrl: string;
  sessionId: string | null;
  caseFile: CaseFile | null;
  progressSteps: ProgressStep[];
  errorMessage: string;
}

const WorkbenchPage: React.FC = () => {
  const [state, setState] = useState<WorkbenchState>({
    appStatus: 'idle',
    videoUrl: '',
    sessionId: null,
    caseFile: null,
    progressSteps: initialProgressSteps,
    errorMessage: ''
  });

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
      setState(prev => ({
        ...prev,
        appStatus: 'results_displayed',
        caseFile: eventData.caseFile as CaseFile
      }));
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


  const handleUrlSubmit = async (url: string) => {
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
          <URLInputForm 
            onSubmit={handleUrlSubmit}
            isSubmitting={state.appStatus === 'submitting'}
          />
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
          <div className="text-center text-red-400">
            <h2 className="text-2xl font-bold">An Error Occurred</h2>
            <p>{state.errorMessage}</p>
          </div>
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