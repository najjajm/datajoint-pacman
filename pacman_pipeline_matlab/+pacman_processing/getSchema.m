function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pacman_processing', 'churchland_analyses_pacman_processing');
end
obj = schemaObject;
end
