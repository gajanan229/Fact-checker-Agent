import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  HelpCircle,
  ShieldCheck,
  ShieldAlert,
  RotateCcw,
} from 'lucide-react';

interface CritiqueDimensionScores {
  factual_accuracy: number;
  tone_respectfulness: number;
  citation_quality: number;
}

interface ResponseClaimVerification {
  claim: string;
  status: string;
  verification_summary: string;
}

interface Critique {
  is_revision_needed: boolean;
  overall_quality_score: number | null;
  quality_scores?: CritiqueDimensionScores | null;
  strengths: string[];
  critical_issues: string[];
  revision_recommendations: string[];
  response_claim_verifications: ResponseClaimVerification[];
  revision_count: number;
  max_revisions: number;
  ran: boolean;
}

interface RedTeamLogProps {
  critique: Critique;
}

type BannerKind = 'skipped' | 'passed' | 'revised' | 'blocked';

interface Banner {
  kind: BannerKind;
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const buildBanner = (critique: Critique): Banner => {
  if (!critique.ran) {
    return {
      kind: 'skipped',
      label: 'Review skipped',
      description: 'The critique stage did not run for this response.',
      icon: <HelpCircle className="w-4 h-4" />,
      color: 'text-gray-400 bg-gray-400 bg-opacity-15 border-gray-500',
    };
  }

  const blocked =
    critique.is_revision_needed && critique.revision_count >= critique.max_revisions;

  if (blocked) {
    return {
      kind: 'blocked',
      label: 'Blocked at revision cap',
      description: `Reviewer still wanted changes after ${critique.revision_count} of ${critique.max_revisions} allowed revisions.`,
      icon: <ShieldAlert className="w-4 h-4" />,
      color: 'text-red-400 bg-red-400 bg-opacity-15 border-red-500',
    };
  }

  if (critique.revision_count > 0) {
    return {
      kind: 'revised',
      label: `Passed after ${critique.revision_count} revision${critique.revision_count === 1 ? '' : 's'}`,
      description: 'Reviewer accepted the response after iterative refinement.',
      icon: <RotateCcw className="w-4 h-4" />,
      color: 'text-yellow-300 bg-yellow-400 bg-opacity-15 border-yellow-500',
    };
  }

  return {
    kind: 'passed',
    label: 'Passed on first review',
    description: 'Reviewer cleared the response without requesting revisions.',
    icon: <ShieldCheck className="w-4 h-4" />,
    color: 'text-green-400 bg-green-400 bg-opacity-15 border-green-500',
  };
};

const formatScore = (value: number | null | undefined): string => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
};

const scoreBarColor = (value: number): string => {
  if (value >= 0.75) return 'bg-green-400';
  if (value >= 0.5) return 'bg-yellow-400';
  return 'bg-red-400';
};

const getStatusStyle = (status: string) => {
  switch (status.toLowerCase()) {
    case 'verified':
      return {
        color: 'text-green-400 border-green-400 bg-green-400 bg-opacity-15',
        icon: <CheckCircle className="w-3.5 h-3.5" />,
        label: 'Verified',
      };
    case 'debunked':
    case 'false':
      return {
        color: 'text-red-400 border-red-400 bg-red-400 bg-opacity-15',
        icon: <AlertCircle className="w-3.5 h-3.5" />,
        label: 'Debunked',
      };
    case 'misleading':
      return {
        color: 'text-yellow-400 border-yellow-400 bg-yellow-400 bg-opacity-15',
        icon: <AlertTriangle className="w-3.5 h-3.5" />,
        label: 'Misleading',
      };
    case 'lacks_context':
    case 'lacks context':
      return {
        color: 'text-yellow-300 border-yellow-300 bg-yellow-300 bg-opacity-10',
        icon: <AlertTriangle className="w-3.5 h-3.5" />,
        label: 'Lacks Context',
      };
    case 'unverifiable':
    default:
      return {
        color: 'text-gray-400 border-gray-500 bg-gray-500 bg-opacity-15',
        icon: <HelpCircle className="w-3.5 h-3.5" />,
        label: 'Unverifiable',
      };
  }
};

interface DimensionRowProps {
  label: string;
  value: number | undefined;
}

const DimensionRow: React.FC<DimensionRowProps> = ({ label, value }) => {
  const numeric = typeof value === 'number' ? Math.max(0, Math.min(1, value)) : 0;
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-gray-300 mb-1">
        <span>{label}</span>
        <span className="font-medium text-gray-200">{formatScore(value)}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-gray-800 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${numeric * 100}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`h-full ${scoreBarColor(numeric)}`}
        />
      </div>
    </div>
  );
};

interface CollapsibleSectionProps {
  title: string;
  count: number;
  tone: 'positive' | 'warning' | 'critical' | 'neutral';
  items: string[];
}

const toneClasses: Record<CollapsibleSectionProps['tone'], string> = {
  positive: 'text-green-300',
  warning: 'text-yellow-300',
  critical: 'text-red-300',
  neutral: 'text-gray-300',
};

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, count, tone, items }) => {
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(prev => !prev)}
        className="w-full px-3 py-2 flex items-center justify-between hover:bg-gray-800 hover:bg-opacity-30 transition-colors"
      >
        <span className={`text-xs font-medium ${toneClasses[tone]}`}>
          {title} <span className="text-gray-500">({count})</span>
        </span>
        {open ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400" />
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-t border-gray-700 px-3 py-2 space-y-2"
          >
            {items.map((item, idx) => (
              <li key={idx} className="text-xs text-gray-300 leading-relaxed flex">
                <span className="text-gray-600 mr-2">•</span>
                <span>{item}</span>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
};

const RedTeamLog: React.FC<RedTeamLogProps> = ({ critique }) => {
  const banner = buildBanner(critique);
  const verifications = critique.response_claim_verifications || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="verifyp-card rounded-xl p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-gray-100">Red Team Log</h3>
        </div>
        {critique.ran && typeof critique.overall_quality_score === 'number' && (
          <span className="text-xs text-gray-400">
            Overall:{' '}
            <span className="text-gray-200 font-medium">
              {formatScore(critique.overall_quality_score)}
            </span>
          </span>
        )}
      </div>

      <div
        className={`flex items-start space-x-2 px-3 py-2 mb-4 rounded-lg border ${banner.color}`}
      >
        <div className="mt-0.5 flex-shrink-0">{banner.icon}</div>
        <div className="text-xs leading-snug">
          <div className="font-semibold">{banner.label}</div>
          <div className="text-gray-400 mt-0.5">{banner.description}</div>
        </div>
      </div>

      {critique.ran && critique.quality_scores && (
        <div className="space-y-3 mb-4">
          <DimensionRow
            label="Factual accuracy"
            value={critique.quality_scores.factual_accuracy}
          />
          <DimensionRow
            label="Tone & respect"
            value={critique.quality_scores.tone_respectfulness}
          />
          <DimensionRow
            label="Citation quality"
            value={critique.quality_scores.citation_quality}
          />
        </div>
      )}

      {critique.ran && verifications.length > 0 && (
        <div className="mb-4">
          <h4 className="text-xs font-medium text-gray-200 mb-2">
            Claims in response ({verifications.length})
          </h4>
          <ul className="space-y-2">
            {verifications.map((entry, idx) => {
              const style = getStatusStyle(entry.status);
              return (
                <li
                  key={idx}
                  className="border border-gray-700 rounded-lg p-2.5 bg-gray-900 bg-opacity-30"
                >
                  <div className="flex items-start justify-between gap-2 mb-1.5">
                    <p className="text-xs text-gray-200 leading-snug flex-1">
                      {entry.claim}
                    </p>
                    <span
                      className={`flex items-center space-x-1 border rounded-full px-2 py-0.5 text-[10px] font-medium flex-shrink-0 ${style.color}`}
                    >
                      {style.icon}
                      <span>{style.label}</span>
                    </span>
                  </div>
                  {entry.verification_summary && (
                    <p className="text-[11px] text-gray-400 leading-relaxed">
                      {entry.verification_summary}
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {critique.ran && (
        <div className="space-y-2">
          <CollapsibleSection
            title="Critical issues"
            tone="critical"
            count={critique.critical_issues.length}
            items={critique.critical_issues}
          />
          <CollapsibleSection
            title="Revision recommendations"
            tone="warning"
            count={critique.revision_recommendations.length}
            items={critique.revision_recommendations}
          />
          <CollapsibleSection
            title="Strengths"
            tone="positive"
            count={critique.strengths.length}
            items={critique.strengths}
          />
        </div>
      )}
    </motion.div>
  );
};

export default RedTeamLog;
