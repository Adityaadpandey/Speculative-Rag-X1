export class ApiError extends Error {}

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (err) {
    throw new ApiError(`network error: ${err.message}`);
  }
  let data;
  try {
    data = await response.json();
  } catch (err) {
    throw new ApiError(`invalid response: ${err.message}`);
  }
  if (!response.ok) {
    throw new ApiError(data.error ?? `request failed: ${response.status}`);
  }
  return data;
}

export function fetchCollections() {
  return request("/api/collections");
}

export function postAnswer({ queryText, language, strategy }) {
  return request("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_text: queryText, language, strategy }),
  });
}

export function getAnswer(requestId) {
  return request(`/api/answer/${requestId}`);
}

export function transcribeAudio({ blob, filename, language }) {
  const formData = new FormData();
  formData.append("file", blob, filename);
  formData.append("language", language);
  return request("/api/transcribe", { method: "POST", body: formData });
}
