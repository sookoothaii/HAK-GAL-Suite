import { type DataSource } from "@/store/useStore";

interface DataSourceItemProps {
  source: DataSource;
}

export const DataSourceItem = ({ source }: DataSourceItemProps) => {
  return (
    <div className="flex items-center gap-2 p-2 neural-border rounded-lg text-xs transition-neural hover:bg-accent/5">
      <div className="w-2 h-2 rounded-full bg-accent/60"></div>
      <span className="flex-1 font-mono text-sm truncate">{source.filename}</span>
      <div className="data-chip text-xs">
        RAG
      </div>
    </div>
  );
};