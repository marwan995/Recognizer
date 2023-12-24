import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "./components/ui/toaster";
import { Routes, Route, useLocation } from "react-router-dom";

import { PagesContainer } from "./pages/PagesContainer/PagesContainer";
import { RecognizePopup } from "./pages/PagesContainer/RecognizePopup";
import { AddPlatePopup } from "./pages/PagesContainer/AddPlatePopup";


const queryClient = new QueryClient();

function App() {
  const location = useLocation();
  const previousLocation = location.state?.previousLocation;
  console.log(previousLocation)
  return (
    <QueryClientProvider client={queryClient}>
            <Routes location={previousLocation || location}>
              <Route path="/" element={<PagesContainer />} />
            </Routes> 
            <Routes>
              {/* this is the popup routs */}
              <Route path="/recognize" element={<RecognizePopup />} />
              <Route path="/add_plate" element={<AddPlatePopup />} />
            </Routes>
            <Toaster />

    </QueryClientProvider>
  );
}

export default App;
