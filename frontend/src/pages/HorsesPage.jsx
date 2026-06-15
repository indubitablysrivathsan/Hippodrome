import { useState, useEffect } from "react";
import { horses } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

export default function HorsesPage() {
  const list = useApi(horses.list);
  const search = useApi(horses.search);
  const detail = useApi(horses.get);
  const aliases = useApi(horses.aliases);
  const ratings = useApi(horses.ratings);
  const currentRating = useApi(horses.currentRating);
  const raceHistory = useApi(horses.races);
  const medical = useApi(horses.medical);
  const treadmill = useApi(horses.treadmill);

  const [id, setId] = useState("");
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState("10");

  useEffect(() => { list.run(); }, []);

  return (
    <div>
      <h2>Horses</h2>

      <Section title="List all horses">
        <Btn onClick={() => list.run()}>Refresh</Btn>
        <Status {...list} />
        {list.data && <Table rows={toRows(list.data)} />}
      </Section>

      <Section title="Search horses">
        <Field label="Query" value={q} onChange={setQ} placeholder="horse name…" />
        <Field label="Limit" value={limit} onChange={setLimit} placeholder="10" />
        <Btn onClick={() => search.run(q, limit)} disabled={!q}>Search</Btn>
        <Status {...search} />
        {search.data && <Table rows={toRows(search.data)} />}
      </Section>

      <Section title="Horse detail">
        <Field label="Horse ID" value={id} onChange={setId} placeholder="e.g. 1" />
        <Btn onClick={() => detail.run(id)} disabled={!id}>Fetch</Btn>
        <Status {...detail} />
        {detail.data && <Table rows={toRows(detail.data)} />}
      </Section>

      {[
        { key: "aliases", label: "Aliases", api: aliases },
        { key: "ratings", label: "Ratings", api: ratings },
        { key: "currentRating", label: "Current Rating", api: currentRating },
        { key: "raceHistory", label: "Race History", api: raceHistory },
        { key: "medical", label: "Medical", api: medical },
        { key: "treadmill", label: "Treadmill", api: treadmill },
      ].map(({ key, label, api }) => (
        <Section key={key} title={label}>
          <Field label="Horse ID" value={id} onChange={setId} placeholder="e.g. 1" />
          <Btn onClick={() => api.run(id)} disabled={!id}>Fetch {label}</Btn>
          <Status {...api} />
          {api.data && <Table rows={toRows(api.data)} />}
        </Section>
      ))}
    </div>
  );
}
