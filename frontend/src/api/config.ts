import { json } from "./client";
import type { SecretsStatusResponse, SecretStatus } from "./generated-types";

export type { SecretsStatusResponse, SecretStatus };

export const getSecretsStatus = () =>
  json<SecretsStatusResponse>("/api/config/secrets");

/** Send only keys whose value should change.
 *  `""` clears; a non-empty string sets; omit to leave untouched. */
export const updateSecrets = (values: Record<string, string>) =>
  json<SecretsStatusResponse>("/api/config/secrets", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
