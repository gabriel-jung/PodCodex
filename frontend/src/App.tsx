import { RouterProvider } from "@tanstack/react-router";
import { router } from "./router";
import { useHydrateAppDefaults } from "./stores";

function App() {
  // Server-owned pipeline defaults; hydrate before pages seed from them.
  useHydrateAppDefaults();
  return <RouterProvider router={router} />;
}

export default App;
