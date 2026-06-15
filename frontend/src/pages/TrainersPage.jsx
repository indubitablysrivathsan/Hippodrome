import { useState, useEffect } from "react";
import { trainers } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

export default function TrainersPage() {
  const list = useApi(trainers.list);
  const detail = useApi(trainers.get);
  const runnerList = useApi(trainers.runners);
  const stats = useApi(trainers.stats);
  const penalties = useApi(trainers.penalties);

  const [id, setId] = useState("");

  useEffect(() => { list.run(); }, []);

  return (
    <div>
      <h2>Trainers</h2>

      <Section title="List all trainers">
        <Btn onClick={() => list.run()}>Refresh</Btn>
        <Status {...list} />
        {list.data && <Table rows={toRows(list.data)} />}
      </Section>

      <Section title="Trainer detail">
        <Field label="Trainer ID" value={id} onChange={setId} placeholder="e.g. 1" />
        <Btn onClick={() => detail.run(id)} disabled={!id}>Fetch</Btn>
        <Status {...detail} />
        {detail.data && <Table rows={toRows(detail.data)} />}
      </Section>

      {[
        { label: "Runners", api: runnerList, fn: () => runnerList.run(id) },
        { label: "Stats", api: stats, fn: () => stats.run(id) },
        { label: "Penalties", api: penalties, fn: () => penalties.run(id) },
      ].map(({ label, api, fn }) => (
        <Section key={label} title={label}>
          <Field label="Trainer ID" value={id} onChange={setId} placeholder="e.g. 1" />
          <Btn onClick={fn} disabled={!id}>Fetch {label}</Btn>
          <Status {...api} />
          {api.data && <Table rows={toRows(api.data)} />}
        </Section>
      ))}
    </div>
  );
}
