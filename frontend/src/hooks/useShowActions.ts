import { useMutation } from "@tanstack/react-query";
import { downloadEpisodes, downloadYouTubeEpisodes, importYouTubeSubs } from "@/api/client";
import { languageToISO } from "@/lib/utils";
import type { ShowMeta } from "@/api/types";
import { useTaskStore } from "@/stores";

export function useShowActions(
  folder: string,
  meta: ShowMeta | undefined,
  opts?: { withSubs?: boolean },
) {
  const { setDownloadTask } = useTaskStore();
  const withSubs = opts?.withSubs ?? true;

  const isYouTube = !!meta?.youtube_url;
  const showLangISO = languageToISO(meta?.language || "") || "en";

  const downloadMutation = useMutation({
    mutationFn: ({ guids, force = false }: { guids: string[]; force?: boolean }) =>
      isYouTube
        ? downloadYouTubeEpisodes(folder, guids, withSubs, showLangISO)
        : downloadEpisodes(folder, guids, force),
    onSuccess: (data) => { setDownloadTask(data.task_id, folder); },
  });

  const importSubsMutation = useMutation({
    mutationFn: ({ ids, lang }: { ids: string[]; lang: string }) =>
      importYouTubeSubs(folder, ids, lang),
    onSuccess: (data) => { setDownloadTask(data.task_id, folder); },
  });

  return { downloadMutation, importSubsMutation, isYouTube, showLangISO };
}
