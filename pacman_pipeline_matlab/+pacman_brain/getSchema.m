function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pacman_brain', 'churchland_analyses_pacman_brain');
end
obj = schemaObject;
end
