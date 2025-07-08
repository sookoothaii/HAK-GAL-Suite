// index.tsx
import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Trash2, Plus, X, Send, ChevronDown, Upload, Loader2 } from "lucide-react";

type Message = {
  type: "user" | "system";
  content: string;
};

const Index = () => {
  const [inputMessage, setInputMessage] = useState("");
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [rightPanelView, setRightPanelView] = useState("rag");
  const [profileEntity, setProfileEntity] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [permanentKnowledge, setPermanentKnowledge] = useState<string[]>([]);
  const [learningSuggestions, setLearningSuggestions] = useState<string[]>([]);
  const [dataSources, setDataSources] = useState<string[]>([]);
  const [ragContext, setRagContext] = useState("No context loaded yet.");
  
  const conversationEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversationHistory]);

  const sendCommandToBackend = async (commandString: string) => {
    if (!commandString.trim() || isLoading) return;

    setIsLoading(true);
    setConversationHistory(prev => [...prev, { type: "user", content: commandString }]);
    setInputMessage("");

    if (commandString.toLowerCase().startsWith("what_is ")) {
        const entity = commandString.slice(8).trim();
        setProfileEntity(entity);
        setRightPanelView("profile");
    } else {
        setRightPanelView("rag");
    }

    try {
      const response = await fetch("http://localhost:5001/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: commandString }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "success") {
        setPermanentKnowledge(data.permanentKnowledge || []);
        setLearningSuggestions(data.learningSuggestions || []);
        
        if (data.chatResponse) {
          setConversationHistory(prev => [...prev, { type: "system", content: data.chatResponse }]);
        } else {
          // Fallback für Befehle ohne direkte Chat-Antwort
          setConversationHistory(prev => [...prev, { type: "system", content: `✅ Command '${commandString}' executed successfully.` }]);
        }
      } else {
        throw new Error(data.message || "An unknown error occurred.");
      }

    } catch (error) {
      console.error("API call failed:", error);
      const errorResponse: Message = {
        type: "system",
        content: `🚨 Error: ${error.message}`,
      };
      setConversationHistory(prev => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = () => sendCommandToBackend(inputMessage);
  const handleLearnAll = () => sendCommandToBackend("learn");
  const handleRetractItem = (item: string) => sendCommandToBackend(`retract ${item}`);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-sans">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-4 h-screen">
        
        <Card className="bg-gray-800/50 border-gray-700 flex flex-col">
          <CardHeader><CardTitle className="text-gray-100">Knowledge Base</CardTitle></CardHeader>
          <CardContent className="flex-1 flex flex-col space-y-4 overflow-hidden">
            <div className="flex-1 flex flex-col">
              <h3 className="text-sm font-medium mb-2 text-gray-400">Permanent Knowledge ({permanentKnowledge.length})</h3>
              <ScrollArea className="flex-1 border border-gray-700 rounded p-2 bg-gray-900/50">
                {permanentKnowledge.length > 0 ? (
                  <div className="space-y-2">
                    {permanentKnowledge.map((item, index) => (
                      <div key={`perm-${index}`} className="flex items-center justify-between text-sm p-2 bg-gray-800 rounded">
                        <span className="flex-1 truncate font-mono text-xs">{item}</span>
                        <Button variant="ghost" size="sm" onClick={() => handleRetractItem(item)} className="ml-2 h-6 w-6 p-0 text-gray-400 hover:text-red-400"><Trash2 className="h-3 w-3" /></Button>
                      </div>
                    ))}
                  </div>
                ) : <p className="text-xs text-gray-500 text-center p-4">No permanent knowledge yet.</p>}
              </ScrollArea>
            </div>
            <div className="flex-1 flex flex-col">
              <h3 className="text-sm font-medium mb-2 text-gray-400">Learning Suggestions ({learningSuggestions.length})</h3>
              <ScrollArea className="flex-1 border border-gray-700 rounded p-2 bg-gray-900/50">
                {learningSuggestions.length > 0 ? (
                  <div className="space-y-2">
                    {learningSuggestions.map((item, index) => (
                      <div key={`learn-${index}`} className="flex items-center justify-between text-sm p-2 bg-gray-800 rounded">
                        <span className="flex-1 truncate font-mono text-xs">{item}</span>
                      </div>
                    ))}
                    <Button variant="outline" size="sm" className="w-full mt-2 border-green-500/50 text-green-400 hover:bg-green-500/10" onClick={handleLearnAll}>Learn All Suggestions</Button>
                  </div>
                ) : <p className="text-xs text-gray-500 text-center p-4">No new facts to learn.</p>}
              </ScrollArea>
            </div>
            <div>
              <h3 className="text-sm font-medium mb-2 text-gray-400">Data Sources</h3>
              <div className="space-y-2">
                {dataSources.map((file, index) => (<div key={index} className="text-xs p-2 bg-gray-800 rounded truncate">{file}</div>))}
                <Button variant="outline" size="sm" className="w-full border-gray-700 hover:bg-gray-700"><Upload className="h-3 w-3 mr-1" />Upload Document</Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800/50 border-gray-700 flex flex-col">
          <CardHeader><CardTitle className="text-gray-100">Interaction Panel</CardTitle></CardHeader>
          <CardContent className="flex-1 flex flex-col overflow-hidden">
            <ScrollArea className="flex-1 mb-4 pr-4">
              <div className="space-y-4">
                {conversationHistory.map((message, index) => (
                  <div key={index} className={`p-3 rounded-lg border ${message.type === 'user' ? 'bg-blue-500/10 border-blue-500/20 text-blue-300' : 'bg-gray-800 border-gray-700'}`}>
                    <div className="flex items-start gap-3">
                      <span className="mt-1">{message.type === 'user' ? '👤' : '🤖'}</span>
                      <pre className="flex-1 whitespace-pre-wrap font-sans text-sm">{message.content}</pre>
                    </div>
                  </div>
                ))}
                {isLoading && <div className="flex justify-center items-center p-4"><Loader2 className="h-6 w-6 animate-spin text-blue-400" /></div>}
                <div ref={conversationEndRef} />
              </div>
            </ScrollArea>
            <div className="flex gap-2 pt-4 border-t border-gray-700">
              <Input value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} placeholder="Enter command or ask a question..." onKeyPress={(e) => e.key === "Enter" && handleSendMessage()} className="flex-1 bg-gray-800 border-gray-600 focus:ring-blue-500" />
              <Button onClick={handleSendMessage} disabled={isLoading} className="bg-blue-600 hover:bg-blue-700 text-white"><Send className="h-4 w-4" /></Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800/50 border-gray-700 flex flex-col">
          <CardHeader><CardTitle className="text-gray-100">{rightPanelView === "rag" ? "RAG Context" : `Profile: ${profileEntity}`}</CardTitle></CardHeader>
          <CardContent className="flex-1 overflow-hidden">
            {rightPanelView === "rag" ? (
              <ScrollArea className="h-full"><div className="text-sm whitespace-pre-wrap font-mono text-gray-400">{ragContext}</div></ScrollArea>
            ) : (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium mb-2 text-gray-400">Explicit Facts</h3>
                  <div className="space-y-1 text-xs font-mono"><div className="p-2 bg-gray-800 rounded">Not yet implemented.</div></div>
                </div>
                <div>
                  <h3 className="text-sm font-medium mb-2 text-gray-400">Derived Properties</h3>
                  <div className="space-y-1 text-xs font-mono"><div className="p-2 bg-gray-800 rounded">Not yet implemented.</div></div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Index;