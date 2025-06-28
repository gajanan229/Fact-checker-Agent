import React from 'react';
import { motion } from 'framer-motion';
import TargetDisplay from './TargetDisplay';
import DraftResponse from './DraftResponse';
import EvidenceLocker from './EvidenceLocker';
import RedTeamLog from './RedTeamLog';
import CollaborationHub from './CollaborationHub';
import FinalizeActions from './FinalizeActions';

interface CaseFileDashboardProps {
  caseFile: any;
  videoUrl: string;
  isRefining: boolean;
  onRefinement: (message: string) => void;
  onCopyToClipboard: () => void;
}

const CaseFileDashboard: React.FC<CaseFileDashboardProps> = ({
  caseFile,
  videoUrl,
  isRefining,
  onRefinement,
  onCopyToClipboard
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-100 mb-2">Case File Complete</h2>
        <p className="text-gray-400">Review the analysis and refine your response</p>
      </div>

      {/* Target Display */}
      <TargetDisplay videoUrl={videoUrl} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Evidence & Process */}
        <div className="lg:col-span-1 space-y-6">
          <EvidenceLocker 
            claims={caseFile.claims} 
            dossier={caseFile.dossier} 
          />
          <RedTeamLog critique={caseFile.critique} />
        </div>

        {/* Right Column - Response & Actions */}
        <div className="lg:col-span-2 space-y-6">
          <DraftResponse 
            response={caseFile.draft_response} 
            isRefining={isRefining} 
          />
          <CollaborationHub 
            onMessage={onRefinement} 
            isRefining={isRefining} 
          />
          <FinalizeActions onCopyToClipboard={onCopyToClipboard} />
        </div>
      </div>
    </motion.div>
  );
};

export default CaseFileDashboard;