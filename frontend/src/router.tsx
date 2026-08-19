import { lazy } from "react";
import {
  createRouter,
  createRoute,
  createRootRoute,
} from "@tanstack/react-router";
import RootLayout from "./pages/RootLayout";
import { ErrorAlert } from "./components/ui/error-alert";

const HomePage = lazy(() => import("./pages/HomePage"));
const ShowPage = lazy(() => import("./pages/ShowPage"));
const EpisodePage = lazy(() => import("./pages/EpisodePage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));

const rootRoute = createRootRoute({
  component: RootLayout,
  // Without this, a throw during render unmounts the tree and leaves a blank
  // page with nothing in the console — the failure mode that made two silent
  // bugs expensive to find. Show the error and offer a way out instead.
  errorComponent: function RouteError({ error, reset }) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
        <ErrorAlert
          error={error}
          details={error instanceof Error ? error.stack : null}
          onRetry={reset}
          className="max-w-2xl"
        />
        <a href="/" className="text-primary hover:underline text-sm">Go home</a>
      </div>
    );
  },
  notFoundComponent: function NotFound() {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
        <p className="text-lg font-medium text-foreground">Page not found</p>
        <a href="/" className="text-primary hover:underline text-sm">Go home</a>
      </div>
    );
  },
});

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: HomePage,
});

const showRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/show/$folder",
  validateSearch: (search: Record<string, unknown>): { tab?: string } => {
    const tab = typeof search.tab === "string" ? search.tab : undefined;
    return tab ? { tab } : {};
  },
  component: function ShowWrapper() {
    const { folder } = showRoute.useParams();
    const { tab } = showRoute.useSearch();
    return <ShowPage folder={decodeURIComponent(folder)} initialTab={tab} />;
  },
});

const episodeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/show/$folder/episode/$stem",
  validateSearch: (search: Record<string, unknown>): { tab?: string } => {
    const tab = typeof search.tab === "string" ? search.tab : undefined;
    return tab ? { tab } : {};
  },
  component: function EpisodeWrapper() {
    const { folder, stem } = episodeRoute.useParams();
    const { tab } = episodeRoute.useSearch();
    return (
      <EpisodePage
        folder={decodeURIComponent(folder)}
        stem={decodeURIComponent(stem)}
        initialTab={tab}
      />
    );
  },
});

export const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  validateSearch: (search: Record<string, unknown>): { tab?: string } => {
    const tab = typeof search.tab === "string" ? search.tab : undefined;
    return tab ? { tab } : {};
  },
  component: SettingsPage,
});

const routeTree = rootRoute.addChildren([homeRoute, showRoute, episodeRoute, settingsRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
