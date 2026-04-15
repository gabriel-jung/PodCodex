import {
  createRouter,
  createRoute,
  createRootRoute,
} from "@tanstack/react-router";
import RootLayout from "./pages/RootLayout";
import HomePage from "./pages/HomePage";
import ShowPage from "./pages/ShowPage";
import EpisodePage from "./pages/EpisodePage";
import SettingsPage from "./pages/SettingsPage";

const rootRoute = createRootRoute({
  component: RootLayout,
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
  component: function EpisodeWrapper() {
    const { folder, stem } = episodeRoute.useParams();
    return (
      <EpisodePage
        folder={decodeURIComponent(folder)}
        stem={decodeURIComponent(stem)}
      />
    );
  },
});

const fileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/file/$path",
  component: function FileWrapper() {
    const { path } = fileRoute.useParams();
    return <EpisodePage audioFilePath={decodeURIComponent(path)} />;
  },
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage,
});

const notFoundRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "*",
  component: function NotFound() {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
        <p className="text-lg font-medium text-foreground">Page not found</p>
        <a href="/" className="text-primary hover:underline text-sm">Go home</a>
      </div>
    );
  },
});

const routeTree = rootRoute.addChildren([homeRoute, showRoute, episodeRoute, fileRoute, settingsRoute, notFoundRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
