import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronRight, ExternalLink, AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';

interface Claim {
  claim_id: number;
  text: string;
  status: string;
}

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

interface EvidenceLockerProps {
  claims: Claim[];
  dossier: { [key: string]: DossierEntry };
}

const EvidenceLocker: React.FC<EvidenceLockerProps> = ({ claims, dossier }) => {
  const [expandedClaims, setExpandedClaims] = useState<number[]>([]);

  const toggleClaim = (claimId: number) => {
    setExpandedClaims(prev => 
      prev.includes(claimId) 
        ? prev.filter(id => id !== claimId)
        : [...prev, claimId]
    );
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'false':
        return 'text-red-400 bg-red-400 bg-opacity-20 border-red-400';
      case 'true':
      case 'verified':
        return 'text-green-400 bg-green-400 bg-opacity-20 border-green-400';
      case 'misleading':
      case 'lacks context':
        return 'text-yellow-400 bg-yellow-400 bg-opacity-20 border-yellow-400';
      default:
        return 'text-gray-400 bg-gray-400 bg-opacity-20 border-gray-400';
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'false':
        return <AlertCircle className="w-4 h-4" />;
      case 'true':
      case 'verified':
        return <CheckCircle className="w-4 h-4" />;
      case 'misleading':
      case 'lacks context':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <AlertCircle className="w-4 h-4" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="verifyp-card rounded-xl p-6"
    >
      <h3 className="text-lg font-semibold text-gray-100 mb-4">Evidence Locker</h3>
      
      <div className="space-y-3">
        {claims.map((claim) => {
          const isExpanded = expandedClaims.includes(claim.claim_id);
          const evidence = dossier[claim.claim_id.toString()];
          
          return (
            <div key={claim.claim_id} className="border border-gray-700 rounded-lg overflow-hidden">
              <button
                onClick={() => toggleClaim(claim.claim_id)}
                className="w-full p-4 text-left hover:bg-gray-800 hover:bg-opacity-30 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-sm font-medium text-gray-300">
                        Claim {claim.claim_id}
                      </span>
                      {evidence && (
                        <div className={`px-2 py-1 rounded-full border text-xs font-medium flex items-center space-x-1 ${getVerdictColor(evidence.verdict)}`}>
                          {getVerdictIcon(evidence.verdict)}
                          <span>{evidence.verdict}</span>
                        </div>
                      )}
                    </div>
                    <p className="text-gray-200 text-sm">{claim.text}</p>
                  </div>
                  <div className="ml-4">
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                </div>
              </button>

              <AnimatePresence>
                {isExpanded && evidence && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 border-t border-gray-700">
                      <div className="mt-4">
                        <h4 className="text-sm font-medium text-gray-200 mb-2">Analysis</h4>
                        <p className="text-sm text-gray-300 leading-relaxed mb-4">
                          {evidence.summary}
                        </p>

                        {evidence.sources.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-gray-200 mb-2">Sources</h4>
                            <ul className="space-y-2">
                              {evidence.sources.map((source) => (
                                <li key={source.id}>
                                  <a
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-sm text-teal-400 hover:text-teal-300 flex items-center space-x-2 group"
                                  >
                                    <span className="truncate">{source.title}</span>
                                    <ExternalLink className="w-3 h-3 flex-shrink-0 group-hover:translate-x-0.5 transition-transform" />
                                  </a>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
};

export default EvidenceLocker;