function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pacman_acquisition', 'churchland_analyses_pacman_acquisition');
end
obj = schemaObject;
end
