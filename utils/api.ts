const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "/api";

export const getApiUrl = (endpoint: string) => {
  // Ensure endpoint starts with /
  const path = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  
  // If it's a relative path (starts with /api), return as is for Vercel
  if (API_BASE_URL.startsWith("/")) {
    return `${API_BASE_URL}${path}`;
  }
  
  // Otherwise, it's an absolute URL (e.g. for local dev or separate hosting)
  return `${API_BASE_URL}${path}`;
};
