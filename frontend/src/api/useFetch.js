import { useState, useEffect, useCallback, useRef } from "react";

/**
 * Generic async data fetcher.
 * @param {Function} fn   - async function that returns data
 * @param {Array}    deps - dependency array (re-fetches when changed)
 */
export function useFetch(fn, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => { if (mounted.current) setData(d); })
      .catch((e) => { if (mounted.current) setError(e.message); })
      .finally(() => { if (mounted.current) setLoading(false); });
    return () => { mounted.current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  const refetch = useCallback(() => {
    setLoading(true);
    setError(null);
    fn()
      .then((d) => { if (mounted.current) setData(d); })
      .catch((e) => { if (mounted.current) setError(e.message); })
      .finally(() => { if (mounted.current) setLoading(false); });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error, refetch };
}

/** Simple paginated fetch wrapper */
export function usePaginated(fn, extraDeps = []) {
  const [limit] = useState(50);
  const [offset, setOffset] = useState(0);
  const { data, loading, error, refetch } = useFetch(
    () => fn(limit, offset),
    [limit, offset, ...extraDeps]
  );

  return {
    data: data?.data ?? [],
    total: data?.total ?? 0,
    limit,
    offset,
    setOffset,
    loading,
    error,
    refetch,
  };
}