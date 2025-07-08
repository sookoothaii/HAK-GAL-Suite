import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import { useStore } from "@/store/useStore";
import { Message } from "./Message";
import { Skeleton } from "@/components/ui/skeleton";

export const InteractionPanel = () => {
  const { 
    conversationHistory, 
    inputMessage, 
    isLoading,
    setInputMessage, 
    sendMessage 
  } = useStore();

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  return (
    <Card className="flex flex-col neural-border shadow-card transition-neural hover:shadow-neural">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="w-6 h-6 rounded bg-gradient-logic flex items-center justify-center">
            <Send className="w-3 h-3 text-primary-foreground" />
          </div>
          Interaction Panel
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        {/* Conversation History */}
        <ScrollArea className="flex-1 neural-border rounded-lg p-4 mb-4 min-h-0">
          {isLoading && (
            <div className="flex flex-col items-center justify-center py-8">
              <div className="relative">
                <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary/30"></div>
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-primary absolute top-0 left-0"></div>
              </div>
              <div className="mt-3 text-center">
                <p className="text-sm font-medium text-primary">Processing Query</p>
                <p className="text-xs text-muted-foreground">Analyzing with SMT/ATP solvers...</p>
              </div>
            </div>
          )}
          {conversationHistory.length === 0 && !isLoading ? (
            <div className="flex flex-col items-center justify-center h-full py-12 text-center">
              <div className="w-16 h-16 rounded-full bg-gradient-neural flex items-center justify-center mb-4">
                <Send className="w-8 h-8 text-primary-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">HAK/GAL Ready</h3>
              <p className="text-sm text-muted-foreground mb-1">
                Start by typing a logical assertion or question below.
              </p>
              <p className="text-xs text-muted-foreground/70">
                Example: "what_is ServerA" or "prove: ∀x. P(x) → Q(x)"
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {conversationHistory.map((message) => (
                <Message key={message.id} message={message} />
              ))}
            </div>
          )}
        </ScrollArea>

        {/* Input Area */}
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Input
              placeholder="Enter logical assertion, query, or command..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              className="neural-border transition-neural focus:shadow-glow font-mono"
              disabled={isLoading}
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <span className="text-xs text-muted-foreground">
                {inputMessage.length > 0 && `${inputMessage.length} chars`}
              </span>
            </div>
          </div>
          <Button 
            onClick={sendMessage} 
            disabled={isLoading || !inputMessage.trim()}
            className="bg-gradient-neural hover:opacity-90 transition-neural px-6"
            size="default"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};