import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { ChevronDown, CheckCircle, Shield, HelpCircle, User } from "lucide-react";
import { type Message as MessageType } from "@/store/useStore";

interface MessageProps {
  message: MessageType;
}

export const Message = ({ message }: MessageProps) => {
  const getMessageStyle = () => {
    switch (message.type) {
      case "success":
        return "bg-success/10 border-success/20 text-success-foreground";
      case "warning":
        return "bg-warning/10 border-warning/20 text-warning-foreground";
      case "unknown":
        return "bg-unknown/10 border-unknown/20 text-unknown-foreground";
      default:
        return "bg-primary/10 border-primary/20 text-primary-foreground";
    }
  };

  const getIcon = () => {
    switch (message.type) {
      case "success":
        return <CheckCircle className="w-4 h-4 text-success animate-fade-in" />;
      case "warning":
        return <Shield className="w-4 h-4 text-warning animate-fade-in" />;
      case "unknown":
        return <HelpCircle className="w-4 h-4 text-unknown animate-fade-in" />;
      default:
        return <User className="w-4 h-4 text-primary animate-fade-in" />;
    }
  };

  const getTypeLabel = () => {
    switch (message.type) {
      case "success":
        return "Verified";
      case "warning":
        return "Warning";
      case "unknown":
        return "Unknown";
      default:
        return "Query";
    }
  };

  return (
    <div className={`neural-border rounded-lg transition-neural animate-fade-in hover:shadow-card ${getMessageStyle()}`}>
      <div className="p-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            {getIcon()}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wide opacity-70">
                {getTypeLabel()}
              </span>
              <div className="h-1 w-1 rounded-full bg-current opacity-30"></div>
              <span className="text-xs opacity-50 font-mono">
                {new Date().toLocaleTimeString()}
              </span>
            </div>
            <p className="text-sm leading-relaxed font-medium">{message.content}</p>
            {message.reasoning && (
              <Collapsible className="mt-3">
                <CollapsibleTrigger asChild>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="p-0 h-auto text-xs hover:bg-transparent opacity-70 hover:opacity-100 transition-neural"
                  >
                    <ChevronDown className="w-3 h-3 mr-1" />
                    Show reasoning & proof
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-3 animate-accordion-down">
                  <div className="logic-display">
                    <div className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wide">
                      Automated Reasoning Trace
                    </div>
                    <pre className="whitespace-pre-wrap text-xs leading-relaxed">
                      {message.reasoning}
                    </pre>
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};