import { useState, useEffect } from "react";
import { races } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

const SUB = [
  { key: "runners", label: "Runners", fn: (api, id) => api.run(id) },
  { key: "dividends", label: "Dividends", fn: (api, id) => api.run(id) },
  { key: "exotics", label: "Exotics", fn: (api, id) => api.run(id) },
  { key: "remarks", label: "Remarks", fn: (api, id) => api.run(id) },
  { key: "penalties", label: "Penalties", fn: (api, id) => api.run(id) },
];

export default function RacesPage() {
  const list = useApi(races.list);
  const detail = useApi(races.get);
  const runners = useApi(races.runners);
  const dividends = useApi(races.dividends);
  const exotics = useApi(races.exotics);
  const remarks = useApi(races.remarks);
  const penalties = useApi(races.penalties);

  const [id, setId] = useState("");

  useEffect(() => { list.run(); }, []);

  const subApis = { runners, dividends, exotics, remarks, penalties };

  return (
    <div>
      <h2>Races</h2>

      <Section title="List all races">
        <Btn onClick={() => list.run()}>Refresh</Btn>
        <Status {...list} />
        {list.data && <Table rows={toRows(list.data)} />}
      </Section>

      <Section title="Race detail by ID">
        <Field label="Race ID" value={id} onChange={setId} placeholder="e.g. 1" />
        <Btn onClick={() => detail.run(id)} disabled={!id}>Fetch</Btn>
        <Status {...detail} />
        {detail.data && <Table rows={toRows(detail.data)} />}
      </Section>

      {SUB.map(({ key, label }) => (
        <Section key={key} title={`Race ${label}`}>
          <Field label="Race ID" value={id} onChange={setId} placeholder="e.g. 1" />
          <Btn onClick={() => subApis[key].run(id)} disabled={!id}>Fetch {label}</Btn>
          <Status {...subApis[key]} />
          {subApis[key].data && <Table rows={toRows(subApis[key].data)} />}
        </Section>
      ))}
    </div>
  );
}
