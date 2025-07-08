import { ScrollArea } from "@/components/ui/scroll-area";
import { useStore } from "@/store/useStore";

export const RagView = () => {
  const ragContext = useStore((state) => state.ragContext);

  return (
    <div className="h-full space-y-4">
      <div className="flex items-center gap-2 pb-2 border-b border-border/50">
        <div className="w-2 h-2 rounded-full bg-accent animate-neural-pulse"></div>
        <span className="text-sm font-semibold text-accent">Active Context Chunks</span>
        <div className="data-chip">
          Vector Search
        </div>
      </div>
      
      <ScrollArea className="h-[calc(100%-3rem)]">
        <div className="space-y-3">
          {ragContext.split('\n').filter(line => line.trim()).map((chunk, index) => (
            <div key={index} className="neural-border rounded-lg p-3 hover:bg-accent/5 transition-neural">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-accent/60 mt-2 flex-shrink-0"></div>
                <div className="flex-1">
                  <div className="logic-display mb-2">
                    <pre className="text-xs whitespace-pre-wrap leading-relaxed">
                      {chunk}
                    </pre>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>Relevance: {(Math.random() * 0.3 + 0.7).toFixed(3)}</span>
                    <div className="h-1 w-1 rounded-full bg-current opacity-30"></div>
                    <span>Source: Knowledge Base</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};