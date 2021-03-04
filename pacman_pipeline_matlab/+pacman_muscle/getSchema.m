function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pacman_muscle', 'churchland_analyses_pacman_muscle');
end
obj = schemaObject;
end
