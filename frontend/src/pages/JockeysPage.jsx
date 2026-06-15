import { useState, useEffect } from "react";
import { jockeys } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

export default function JockeysPage() {
  const list = useApi(jockeys.list);
  const detail = useApi(jockeys.get);
  const rides = useApi(jockeys.rides);
  const penalties = useApi(jockeys.penalties);
  const stats = useApi(jockeys.stats);

  const [id, setId] = useState("");

  useEffect(() => { list.run(); }, []);

  return (
    <div>
      <h2>Jockeys</h2>

      <Section title="List all jockeys">
        <Btn onClick={() => list.run()}>Refresh</Btn>
        <Status {...list} />
        {list.data && <Table rows={toRows(list.data)} />}
      </Section>

      <Section title="Jockey detail">
        <Field label="Jockey ID" value={id} onChange={setId} placeholder="e.g. 1" />
        <Btn onClick={() => detail.run(id)} disabled={!id}>Fetch</Btn>
        <Status {...detail} />
        {detail.data && <Table rows={toRows(detail.data)} />}
      </Section>

      {[
        { label: "Rides", api: rides, fn: () => rides.run(id) },
        { label: "Penalties", api: penalties, fn: () => penalties.run(id) },
        { label: "Stats", api: stats, fn: () => stats.run(id) },
      ].map(({ label, api, fn }) => (
        <Section key={label} title={label}>
          <Field label="Jockey ID" value={id} onChange={setId} placeholder="e.g. 1" />
          <Btn onClick={fn} disabled={!id}>Fetch {label}</Btn>
          <Status {...api} />
          {api.data && <Table rows={toRows(api.data)} />}
        </Section>
      ))}
    </div>
  );
}
