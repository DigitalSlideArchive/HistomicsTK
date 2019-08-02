import { wrap } from 'girder/utilities/PluginUtils';
import { AccessType } from 'girder/constants';
import { restRequest } from 'girder/rest';
import events from 'girder/events';
import ItemListWidget from 'girder/views/widgets/ItemListWidget';

import '../stylesheets/views/itemList.styl';

wrap(ItemListWidget, 'render', function (render) {
    const root = this;

    render.call(this);

    function adjustView(settings) {
        if (!settings || !settings['histomicstk.quarantine_folder']) {
            return;
        }
        root.$el.find('.g-item-list-entry').each(function () {
            var parent = $(this);
            parent.remove('.g-histomicstk-quarantine');
            parent.append($('<a class="g-histomicstk-quarantine"><span>Q</span></a>').attr({
                'g-item-cid': $('[g-item-cid]', parent).attr('g-item-cid'),
                title: 'Move this item to the quarantine folder'
            }));
        });
    }

    function quarantine(event) {
        const target = $(event.currentTarget);
        const cid = target.attr('g-item-cid');
        const item = root.collection.get(cid);
        restRequest({
            type: 'PUT',
            url: 'HistomicsTK/quarantine/' + item.id,
            error: null
        }).done((resp) => {
            events.trigger('g:alert', {
                icon: 'ok',
                text: 'Item quarantined.',
                type: 'success',
                timeout: 4000
            });
            root.trigger('g:changed');
            if (root.parentView && root.parentView.setCurrentModel && root.parentView.parentModel) {
                root.parentView.setCurrentModel(root.parentView.parentModel, {setRoute: false});
            } else {
                target.closest('.g-item-list-entry').remove();
            }
        }).fail((resp) => {
            events.trigger('g:alert', {
                icon: 'cancel',
                text: 'Failed to quarantine item.',
                type: 'danger',
                timeout: 4000
            });
        });
    }

    if (this.accessLevel >= AccessType.WRITE) {
        if (!this._htk_settings) {
            restRequest({
                type: 'GET',
                url: 'HistomicsTK/settings'
            }).done((resp) => {
                this._htk_settings = resp;
                adjustView(this._htk_settings);
            });
        } else {
            adjustView(this._htk_settings);
        }
        this.events['click .g-histomicstk-quarantine'] = quarantine;
        this.delegateEvents();
    }
});
