import { useState, useEffect } from "react";
import { schema } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

export default function SchemaPage() {
  const tables = useApi(schema.tables);
  const columns = useApi(schema.columns);
  const stats = useApi(schema.stats);
  const activity = useApi(schema.recentActivity);

  const [table, setTable] = useState("");
  const [limit, setLimit] = useState("10");

  useEffect(() => {
    tables.run();
    stats.run();
    activity.run(10);
  }, []);

  return (
    <div>
      <h2>Schema / Stats</h2>

      <Section title="Tables">
        <Btn onClick={() => tables.run()}>Refresh</Btn>
        <Status {...tables} />
        {tables.data && <Table rows={toRows(tables.data)} />}
      </Section>

      <Section title="Columns for table">
        <Field label="Table name" value={table} onChange={setTable} placeholder="e.g. races" />
        <Btn onClick={() => columns.run(table)} disabled={!table}>Fetch</Btn>
        <Status {...columns} />
        {columns.data && <Table rows={toRows(columns.data)} />}
      </Section>

      <Section title="Stats">
        <Btn onClick={() => stats.run()}>Refresh</Btn>
        <Status {...stats} />
        {stats.data && <Table rows={toRows(stats.data)} />}
      </Section>

      <Section title="Recent Activity">
        <Field label="Limit" value={limit} onChange={setLimit} placeholder="10" />
        <Btn onClick={() => activity.run(limit)}>Fetch</Btn>
        <Status {...activity} />
        {activity.data && <Table rows={toRows(activity.data)} />}
      </Section>
    </div>
  );
}
