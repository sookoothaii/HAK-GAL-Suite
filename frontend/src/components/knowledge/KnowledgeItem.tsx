import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { useStore, type KnowledgeItem as KnowledgeItemType } from "@/store/useStore";

interface KnowledgeItemProps {
  item: KnowledgeItemType;
}

export const KnowledgeItem = ({ item }: KnowledgeItemProps) => {
  const deleteKnowledge = useStore((state) => state.deleteKnowledge);

  return (
    <div className="group flex items-center justify-between p-3 neural-border rounded-lg text-xs transition-neural hover:bg-primary/5 animate-fade-in">
      <div className="flex items-center gap-2 flex-1">
        <div className="w-2 h-2 rounded-full bg-primary/60 animate-neural-pulse"></div>
        <span className="flex-1 font-mono text-sm leading-relaxed">{item.content}</span>
      </div>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => deleteKnowledge(item.id)}
        className="h-7 w-7 p-0 ml-2 opacity-0 group-hover:opacity-100 transition-neural hover:bg-destructive/10 hover:text-destructive"
      >
        <Trash2 className="h-3 w-3" />
      </Button>
    </div>
  );
};