import { Button } from "@/components/ui/button";
import { Plus, X } from "lucide-react";
import { useStore, type KnowledgeItem } from "@/store/useStore";

interface LearningSuggestionItemProps {
  item: KnowledgeItem;
}

export const LearningSuggestionItem = ({ item }: LearningSuggestionItemProps) => {
  const { addToKnowledge, discardSuggestion } = useStore();

  return (
    <div className="group flex items-center justify-between p-3 neural-border rounded-lg text-xs transition-neural hover:bg-secondary/5 animate-fade-in">
      <div className="flex items-center gap-2 flex-1">
        <div className="w-2 h-2 rounded-full bg-secondary/60 animate-neural-pulse"></div>
        <span className="flex-1 font-mono text-sm leading-relaxed">{item.content}</span>
      </div>
      <div className="flex gap-1 ml-2 opacity-0 group-hover:opacity-100 transition-neural">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => addToKnowledge(item)}
          className="h-7 w-7 p-0 hover:bg-success/10 hover:text-success transition-neural"
          title="Add to Knowledge Base"
        >
          <Plus className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => discardSuggestion(item.id)}
          className="h-7 w-7 p-0 hover:bg-destructive/10 hover:text-destructive transition-neural"
          title="Discard Suggestion"
        >
          <X className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
};